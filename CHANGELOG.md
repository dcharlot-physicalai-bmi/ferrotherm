# Changelog

## Unreleased

No Rust source changed, so no crate version moved. What changed is the **visual editor**, which is
shipped from `docs/` to the site rather than to crates.io, and the two gates that now hold it to the
library.

### The editor could say six of the nine constraints the model layer has

`model` has nine: `not_equal`, `equal`, `fix`, `cardinality`, `at_most`, `at_least`, `exactly_one`,
`at_most_one`, `all_different`. The C ABI reaches all nine — `ft_model_close` takes a kind code, and
kinds 3, 4 and 5 are the last three. The node editor called kinds 0, 1 and 2 and nothing else, so
`all_different` — the constraint the MCP tool description itself calls *the workhorse of assignment,
scheduling, colouring and puzzles* — could not be stated in the visual environment at all. The page's
own shipped example was three pairwise inequalities doing by hand what one `all_different` says.

Nothing caught it because nothing looked. `scripts/check-parity.sh` asks whether every C ABI symbol
reaches every **language binding**; the editor is not a language binding, so it sat outside the one
gate that would have asked. A missing node type is not a build error anywhere. It is simply a thing
a modeller cannot say.

Now stated, and held there:

- **`all_different`, `exactly_one`, `at_most_one`** are node types, wired through `ft_model_var` /
  `ft_model_lit` and `ft_model_close`.
- **Counting constraints grew their ports.** They drew four, which is what a node with four drawn
  ports can hold and not what "at most two of these nine shifts" needs. They now build the literal
  list and close it, which has no arity ceiling. Documents and scripts naming the old `a`/`b`/`c`/`d`
  ports still load: a positional port asked for by name is a request for *a* port, and the next free
  one is the honest answer.
- **Every constraint carries `soft`.** Zero is a rule; a positive number is a preference priced at
  that weight, which the solver may trade away while the answer stays feasible. That is half of
  modelling and the editor could not say it. The report now distinguishes `broken:` from `traded:`
  and prices the trade, because printing both under one word teaches that "broken" and "not an
  answer" are the same thing, which is exactly the distinction soft constraints exist to draw.
- **Variables carry an encoding**, one-hot or domain-wall. Binary is deliberately not offered: the
  compiler refuses a binary-encoded variable in *any* constraint or objective, so the option would
  be a guaranteed refusal rather than a capability. That exemption is written down in the gate.
- **The compiler's `caveats` reach the report.** They were computed and dropped. One of them — that
  `at_most` with k = 1 buys a slack variable `at_most_one` does not need — is now volunteered.

### The model an agent writes and the picture a person edits are one document

`window.ferrotherm.fromModel(m)` lays out the same JSON `ferrotherm_solve` and `POST /v1/solve`
take, and `toModel()` reads it back. Positions are computed, because the model has none. There are
`Copy model` / `Paste model` buttons beside the graph-JSON pair, and `Copy link` puts the model in
the URL **fragment** — the half a browser does not send to a server — so opening somebody's model
does not put their problem in a log. An open editor follows a pasted link rather than ignoring it,
with the previous graph one Undo away.

Two things it refuses rather than approximates, both by name:

- a counting constraint whose literals name **different values** has no drawing, since a counting
  node holds one value for all of them. It still runs through HTTP and MCP, and the refusal says so.
- a variable **no constraint or objective mentions** is not reachable from Solve, so `toModel` would
  drop it. Caught on the way in rather than lost on the way out.

### Two gates, each of which had to fail first

- **`scripts/check-editor-parity.sh`** — every `Constraint` variant is reachable from a node, every
  `Encoding` is offered or exempt *with a reason*, every constraint can be priced. Its `--selftest`
  removes `all_different` from a copy of the page and demands a failure; the first version of that
  self-test **passed the damaged editor**, because an unconditional `EDITOR=docs/graph.html` at the
  top of the script overwrote the path the self-test was pointing at it. A gate that has only ever
  passed is indistinguishable from a gate that cannot fail.
- **`scripts/check-editor-model.sh`** — one model through the API and through `fromModel`, comparing
  the **compiled** size and feasibility across six models covering every constraint kind and both
  objective shapes. Parity is a check on vocabulary and would pass an editor that silently drops a
  `k`: such an editor draws every node type, runs, and answers a different question. Its selftest
  damages `fromModel` to drop a `k` and demands the failure — necessary, because this gate reads its
  numbers out of a report by regex, and a regex that stops matching reports −1 on both sides and
  calls it agreement.

Three assertions in the first draft of the editor tests were **wrong about the library** and the
library was right: `cardinality(k=1)` is not more expensive than `exactly_one` (they compile to
identical graphs, and the compiler declines to caveat either); domain-wall does not save a spin
end-to-end (it saves one *per variable in isolation* — 3 against 4 — and loses across a pairwise
constraint, 9 against 8, because the lowering pays ancillas to return to quadratic); and a binary
encoding produces no caveat from the editor because it produces no *model* — it is refused before
compiling. All three now assert what is true, including the losses.

## 0.30.0

### The four gaps the toolchain survey named, closed

TR-2026-47 surveyed the software layer of thermodynamic computing across five language ecosystems
and placed this stack in it, gaps first. Four algorithms were named as missing. All four are here.

**Isoenergetic cluster moves** (`icm`) are the one that mattered most, because parallel tempering
with them is the baseline the Ising-machine literature measures against — having PT and not the
cluster moves is having the name of the baseline. The move takes a connected component of the
disagreement between two replicas and flips it in **both at once**, and it is always accepted
because the pair's energy is preserved *exactly*: a boundary edge joins a site where the replicas
disagree to one where they agree, so its contribution `−J(a_i·a_j + b_i·b_j)` is zero on both sides
of the flip. That is an equality, so it is asserted as one, to `1e-9`, on every move.

Measured against the identical ladder with the move switched off — same seeds, same budget, two
replica sets either way:

| lattice | spins | ICM wins | loses | ties | mean ΔE |
|---|---|---|---|---|---|
| 8×8 | 64 | 0 | 0 | **20** | 0.00 |
| 12×12 | 144 | 1 | 1 | 18 | 0.00 |
| 16×16 | 256 | 9 | 0 | 11 | −1.80 |
| 20×20 | 400 | 15 | 0 | 5 | −4.00 |
| 24×24 | 576 | **19** | 0 | 1 | **−8.00** |

The advantage grows with size, which is the literature's actual claim rather than "ICM is faster".
The first version of the unit test for this ran at 8×8 and asserted `wins > losses` over **0 and
0** — a comparison where both arms tie on every instance is not a weak test, it is a test of
nothing, and it read as passing. It runs at 16 now, and `examples/icm_scaling` measures the trend.

**Simulated quantum annealing** (`sqa`) is Suzuki–Trotter path-integral Monte Carlo: `M` classical
slices, each carrying the problem at `J/M`, coupled along the Trotter direction at
`J⊥ = −(M/2β)·ln tanh(βΓ/M)`. Two things it does not hide. `Γ` is never annealed to zero, because
`tanh(0) = 0` makes `J⊥` infinite — that is a division, not a strong classical limit. And **one
Trotter slice is not the classical limit**: with `M = 1` a site's two Trotter neighbours are itself,
so the term becomes a constant that suppresses every flip equally. `M = 1` therefore drops the term,
which *is* classical annealing, and is the control the quantum arm is compared against at **matched
work** — one slice at four sweeps per step against four slices at one.

**Goemans–Williamson rounding** (`sdp::goemans_williamson`) was the cheapest and the most
embarrassing: all the semidefinite machinery was already here and used only from the dual side, for
a bound. The primal side is the solution, and it carries the only worst-case guarantee in max-cut.
Verified against optima *proved* by branch-and-bound on 24 instances where the hypothesis holds.
`Rounding::guaranteed` is false on most instances anyone cares about — 0.87856 is stated for
non-negative edge weights, which here means non-positive couplings and no fields — and a guarantee
field that were always true would be a lie in the shape of a guarantee.

**Higher-order models** (`hubo`) are now solved without quadratising. `ΔE_i = 2·Σ_{T∋i} w_T·Π s_j`
costs `O(terms containing i)`, the same shape as the pairwise update with degree replaced by
term-incidence — so a `k`-body model is not harder to *sample*, only harder to put on *pairwise
hardware*, and the reduction pass conflates those two whenever the hardware is not the constraint.
`Outcome::ancillas_avoided` reports what the reduction would have cost. Verified against exhaustive
enumeration over `2¹⁴`.

Five C ABI symbols (`ft_gw_round`, `ft_gw_guaranteed`, `ft_icm`, `ft_icm_moves`, `ft_sqa`) on
header, Python, Zig and Julia, plus `cluster`, `quantum` and `goemans_williamson` methods on HTTP
and MCP. **118 C ABI symbols reach every surface.**

### A "formatting-only" lint fix changed two constants

Clippy asked for `unusual_byte_groupings` on seven hex literals and I regrouped all seven. Two of
them I regrouped **wrong**: `0x4_0B0` is four hex digits and I wrote five, `0x0004_00B0`, turning
16560 into 262320; the same slip took `0x60E_3` from 24803 to 396848. Both are RNG stream constants,
so nothing failed to compile and nothing looked different — a sampler simply ran on a different
stream, and a `hubo` test that had been asserting the exact optimum on all twelve seeds stopped
finding it on one.

Two things came out of that. The constants are restored and **checked numerically** rather than by
eye: `int(a) == int(b)` for every pair, which takes a second and is the only way to know a
regrouping preserved a value. And the test itself was wrong to be that brittle — it asserted a
stochastic sampler as if it were deterministic, so an unrelated edit anywhere in the RNG could trip
it. It now asserts **soundness on every seed** (an energy below the exhaustive minimum would mean
the model and the enumeration disagree about what energy *is*) and **quality as a majority**. Its
failure message was also describing the opposite failure from the one the assertion catches.

**Still open, and named rather than left invisible:** `hubo` has no C ABI. It needs a
model-*building* surface rather than a solver call on an existing handle, which is a different shape
from the other four, and half-doing it is how a capability ends up reachable from one place — the
exact failure 0.25.0 was about.

## 0.29.0

### G11's best-known max-cut has stood for twenty-five years. It is optimal, and this proves it.

| instance | torus | odd dual faces | best known (a lower bound) | **upper bound** | verdict |
|---|---|---|---|---|---|
| G11 | 8×100 | 434 | 564 | **564** | **the bracket closes: 564 is OPTIMAL** |
| G12 | 16×50 | 394 | 556 | **558** | the optimum is in [556, 558] |
| G13 | 32×25 | 384 | 582 | **583** | the optimum is in [582, 583] |

G-set publishes one number per instance and it is always the same kind of number: a best cut
**found**. That is a lower bound — somebody achieved it — and it can never say whether anything
better exists. These are the other end.

A torus is not a plane, so `planarcut::solve` refuses G11 and is right to. But the dual argument
needs only **faces**, and an embedding on any surface has them. Run the same reduction on a toroidal
embedding and the arithmetic is identical; what changes is what the answer means. On a torus the
cycle space of the dual is four times the cut space, so the relaxation ranges over sets that are not
cuts, and its optimum can only be an **upper bound**. Every cut is such a subgraph, so a maximum
over the larger set is at least the maximum over the smaller one — which is the whole proof.

The grid dimensions are **recovered from the edge list**, not assumed. `planar::torus_grid_of` tries
every factorisation and requires the entire edge set to match; agreement on all 1,600 edges is a
statement about the graph rather than a guess about the file.

### The bound says when it is not a bound

`bound_on_surface` reports `attained`: whether the relaxation's optimum is itself a genuine cut. An
even subgraph of the dual is a cut **exactly when it two-colours the graph**, which was already the
self-check on the exact path — so the same walk that catches a broken reduction on the sphere
decides, above it, whether the answer is a proof or a bound. On the sphere it always holds, and a
test asserts that, because otherwise "attained" would be measuring the reduction rather than the
topology.

The toroidal test requires three things against branch and bound: the bound is never below the
proved maximum; when attained it equals it exactly; and it is attained on **some but not all**
instances. Without that last clause the flag could be a constant and the suite would not notice.

### What is not here, and is not claimed

**Exact genus-1 max-cut is not implemented.** I expected it to be four planar subproblems. It is
not: Barahona's polynomial algorithm for arbitrary orientable surfaces needs modular arithmetic over
a generalized-nested-dissection solve, and the 2026 survey of toroidal max-cut offers a *heuristic*
plus exactly this relaxation as the upper bound. The recollection was wrong and the correction is
worth more than the guess would have been — checking it before building saved a fortnight of
building the wrong thing.

Two C ABI symbols (`ft_toroidal_bound`, `ft_toroidal_attained`) on header, Python, Zig and Julia,
plus `toroidal_bound` on HTTP and MCP. **113 C ABI symbols reach every surface.**

## 0.28.0

### Exact max-cut at 10,000 spins

Everything else in this crate searches. This does not. Max-cut is NP-hard **in general** and
polynomial **on a planar graph**, and the difference is a theorem rather than an engineering margin.

| grid | spins | odd dual faces | **exact cut** | breakout local search | BLS short by |
|---|---|---|---|---|---|
| 10×10 | 100 | 42 | **75** | 74 | 1.33% |
| 20×20 | 400 | 180 | **270** | 268 | 0.74% |
| 40×40 | 1,600 | 742 | **1,115** | 1,089 | 2.33% |
| 60×60 | 3,600 | 1,746 | **2,561** | 2,486 | 2.93% |
| 100×100 | 10,000 | 4,848 | **7,040** | 6,864 | 2.50% |

For scale: branch and bound with a certified SDP bound *proves* 76 spins (0.26.0). This proves
10,000. The control is breakout local search — the record holder on most of G-set — precisely
because it is good: a strong heuristic falling 2.5% short is the demonstration that structure beats
search when the structure is there.

The chain is `matching` → `planar` → `planarcut`. Fix an embedding; its faces are the vertices of
the dual; **a cut in `G` is a cycle in `G*`**, because a cut crosses every cycle evenly and the
cycles of the dual are the face boundaries. Complementing turns "maximise an even subgraph" into
"minimise a `T`-join", and a minimum-weight `T`-join is a minimum-weight perfect matching over the
odd-degree dual vertices under shortest-path distances. Negative weights are handled exactly rather
than excluded: a negative edge is taken into the join up front and the parity requirement at both
its endpoints flipped to pay for it.

### It checks itself twice, and refuses rather than reports

Five pieces — blossom, embedding, dual, `T`-join, two-colouring — none of which raises anything when
subtly wrong. So two invariants that come free are asserted on every run: **the recovered edge set
must two-colour** (the dual argument says it is a cut, so walking the graph and flipping across cut
edges must never contradict itself), and **two disjoint computations of the cut must agree**
(`W − w(F)` from the join, against what the recovered state actually cuts). Either failing returns
an error and no number.

It also refuses four different things, and says which, because they are four different instructions:
a **field** makes it a different problem (the standard reduction adds an apex vertex, which is not
planar); a **non-planar** graph is a fact about the instance — a periodic lattice is a torus;
a **cut vertex** is an instruction to split, since max-cut decomposes exactly across biconnected
components; **non-integral weights** are refused rather than rounded, because rounding here moves
the optimum rather than the last digit.

### The blossom hung, and then was quietly wrong 400 vertices later

The first version of `matching` was a paraphrase of the standard primal-dual blossom and it did not
terminate. Four divergences, each individually plausible: **unlabelled and outer nodes shared a
value**, so the alternating tree could not tell a free vertex from one it had already reached;
`augment` walked `slack` where the structure threads through `pa`; slacks were taken between node
pairs rather than between the *representative real edges* that blossom nodes stand for; and
`add_blossom` reversed the whole petal list instead of the tail after the base.

Rewritten faithfully with the edge triple explicit, it then failed differently: `hi − cost` made the
maximum-cost edge weight exactly **0**, which the solver reads as "no edge", so a complete graph
with repeated costs lost enough edges to destroy its perfect matching — and correctly reported that
none existed. The `+ 1` that fixes it is load-bearing and cancels, because every perfect matching
has the same edge count.

And then one more, which the brute-force test could not see: `augment` passed the real vertex where
the reference passes its **containing blossom**. Invisible below about 400 vertices, where blossoms
rarely nest; fatal at 1,600. Exhaustive enumeration stops at about 14. So a **planted** optimum was
added — pair the vertices at cost 0, price everything else at 1 — which is checkable at 500
vertices and would have caught it on the first run.

### Verification

`matching` is checked against exhaustive enumeration over 200 random instances plus negative costs,
forced blossoms, and the planted optimum at 500 vertices. `planar` asserts **Euler's formula** on
every embedding it returns, refuses `K5` and `K3,3` by name, refuses a periodic lattice, and checks
that the rotation at each vertex is a permutation of the real neighbours. `planarcut` agrees with
`branch::solve` — a completely different argument, enumeration in the spin domain — on 60 random
planar instances, and no heuristic is ever allowed to beat it.

Four C ABI symbols (`ft_planar_cut`, `ft_planar_faces`, `ft_planar_odd_faces`, `ft_planar_error`)
on header, Python, Zig and Julia, plus `exact_planar` on HTTP and MCP. **111 C ABI symbols now
reach every surface.** The error text crosses too: a bare NaN would collapse four instructions into
"it did not work".

## 0.27.0

### Breakout local search, and the head-to-head this crate did not have

`tabu` is the baseline a new max-cut heuristic has to beat. `bls` is the thing that beat it: Benlic
& Hao's breakout local search improved the best-known cut on **33** of 71 G-set instances and
matched it on 35 more. A stack reporting a max-cut number without being able to run this is
reporting a number from the wrong decade.

It is descent plus a perturbation that thinks. The descent has **no tabu list at all** — the paper
is explicit that this is the point rather than an omission, arguing that diversification during
descent is what tabu search and annealing both get wrong and that the compromise matters only once a
local optimum is reached. What happens there is the algorithm: three move sets (highest-gain spin;
highest-gain from each side; uniformly random), a jump magnitude `L` that grows **only when a
descent lands on the same optimum as last time**, and a directed-versus-random mix following
`P = max(e^(−ω/T), P0)` in the count of consecutive non-improving descents.

`examples/maxcut_shootout` runs three solvers on one instance at the same number of spin flips,
8 seeds each:

| instance | degree | parallel tempering | tabu search | **breakout local search** | best known |
|---|---|---|---|---|---|
| G11 | 4.0 | 556 | 560 | **562** | 564 |
| G14 | 11.7 | 3045 | **3057** | 3054 | 3064 |
| G1 | 47.9 | 11612 | 11622 | **11624** | 11624 |

BLS matches the world best-known cut on G1 and wins two of three — the result the literature
predicts, and the first time this crate could check it.

### The first version of that table was measured in the wrong unit

It gave parallel tempering `budget` flips and gave tabu and BLS `budget / n`, because their *ledger*
charge is `n` move evaluations per flip — and a budget in ledger samples is not a budget in flips.
At n = 800 that handed the deterministic searches **500 flips against tempering's 320,000**, and the
table read as a clean sweep for tempering on every instance, with BLS last. The tell was in a column
that was already being printed: 12 to 26 descents. A search that visits twenty local optima on an
800-node graph has not run.

The asymmetry that remains is stated rather than hidden: tempering pays `O(degree)` to make a flip
where tabu and BLS pay `O(n)` to **choose** one, because neither has the Fiduccia–Mattheyses bucket
structure the paper uses. A matched-flip table flatters the deterministic searches on quality and
says nothing about what their flips cost. There is no single budget fair to both, and naming the
asymmetry beats pretending there is.

### The published pseudo-code is ambiguous, and that is a parameter

Algorithm 1 sets `ω ← 0` on an improvement; it also sets `ω ← 0` on the stagnation reset; and
Algorithm 2 branches on `ω = 0` to apply the *random* perturbation — whose own comment says it is
for the stagnation case. By the time the perturbation procedure sees `ω`, the two paths are
indistinguishable. `Params::random_after_improvement` carries both readings, defaulting to the
pseudo-code as written, and a test asserts they produce measurably different searches. A parameter
is the honest way to carry an ambiguity in a source; picking one silently is not.

### An off-by-one the Rust test could not see

`iterations_run` came back as 20,001 against a 20,000 budget. The `M2` perturbation applies **two**
moves and the budget was only checked before the pair, so the second went through after the first
had reached the ceiling. The Rust test asserted the budget was spent — on one ferromagnet, at one
budget, where the pair never straddled the boundary. The HTTP test failed instead, on a different
instance at a different budget. The test now sweeps three budgets and six instances, which is what
it should have done to be a test of the property rather than of an example.

### Reachable everywhere on the same day

Four C ABI symbols (`ft_bls`, `ft_bls_descents`, `ft_bls_iterations`, `ft_bls_max_jump`) on the
header, Python, Zig and Julia, plus `"method": "breakout"` on HTTP and MCP with the full
perturbation breakdown in the reply. 107 C ABI symbols now reach every surface that should have
them. `descents` is exported everywhere because it is the number that says whether the algorithm
ran: a handful means the budget was spent inside one basin, and the energy alone would not say so.

## 0.26.0

### A certified SDP bound inside the branch-and-bound tree, on by default

`exact_reach` measured where the cheap bound runs out — 76 spins at mean degree 6, 44 at mean
degree 22 — and its reading said the dense column is where an SDP bound *inside* the tree, rather
than only at the root, would start to pay. It does, and by more than the argument suggested.

The residual at any node is a real graph: the free spins, `λ` as their fields, the free–free
couplings. So [`sdp::certified`] applies with no special case, and the bound it returns is sound by
the same weak-duality argument, verified by the same completed Cholesky — **re-verified before it is
allowed to discard a subtree**, because an unsound bound here does not raise anything. It throws
away the branch holding the optimum and the run still reports `proved_optimal`, in a field
indistinguishable from a correct proof.

| family | mean degree | cheap bound proves | with the SDP bound | nodes at the cheap ceiling |
|---|---|---|---|---|
| sparse | 6.0 | 76 spins | **84 spins** | 8,277,603 → 156,793 (53×) |
| dense | 22.1 | 44 spins | **52 spins** | 12,173,789 → 192,501 (63×) |

### The first measurement of it said it did nothing, and that was a property of the setting

At depth 2 the bound fired about twenty times per instance-set, pruned nought to four, and left the
node count unchanged on 17 of 19 sizes — a clean negative, reported as a fact about the method.
Depth 2 is at most seven nodes. Even a perfect bound there removes a constant fraction of a tree
whose cost is set exponentially deeper down, so the setting could not have shown anything.

`examples/sdp_in_tree` sweeps it instead of picking one:

| spins | cheap | d4 | d8 | d12 | d16 | saturates |
|---|---|---|---|---|---|---|
| 32 | 94,809 | 68,769 | 17,465 | 17,465 | 17,465 | d8 |
| 36 | 242,943 | 160,381 | 13,963 | 1,731 | 1,731 | d12 |
| 40 | 2,181,007 | 1,869,399 | 379,181 | 17,231 | **2,451** | d16 |

**It saturates because the tree closes above that depth.** Once the bound is on, no branch survives
past that level, so a deeper setting has nothing left to visit — which means depth was never the
real control. `sdp_min_free` (a subtree over a handful of spins is enumerated in fewer operations
than one Cholesky) and `sdp_max_free` (the Cholesky is `O(m³)` and runs several times per bound, so
an unbounded ceiling turns a large graph into minutes **per node** and looks like a hang) are. The
default depth is set past where any of it bites.

The `sdp fired` column reads prunes-over-calls, in both examples and in the HTTP reply. Calls alone
would say the bound **ran**, which is not the same as saying it **helped** — and a bound that fires
three hundred times and cuts nothing is three hundred wasted Choleskys.

### A gate that could report a false failure

`check-parity.sh` tested EXEMPT-table membership with `printf '%s\n' "${symbols[@]}" | grep -qx
"$sym"`. On one loaded run it reported `ft_gpu_w` and `ft_cert_passed` as exported by nothing —
both exist, both were untouched, and the very next run passed. `grep -q` exits at the first match,
and whichever `grep` is on `PATH` is free to be raced by that.

**A gate that can report a false failure is worse than no gate**, because people learn to re-run it
instead of believing it, and then the next real failure gets re-run too. Replaced with a shell
`case` over the joined array: no subprocess, no pipeline, deterministic. Five consecutive runs under
load average 109, all green.

### A gate that skipped every step and still printed a summary

The local release script used `${name^^}` to upper-case a step label. That is bash 4; macOS ships
bash 3.2, where it is a syntax error that kills the loop. Everything after the first step was
skipped, and the script still printed `EXAMPLES ran=18 failed=1` — which reads as *one example
broke*, not as *the tests, the docs, the parity check and the answer check never ran*. Caught by
reading the log for the exit codes instead of the summary line. Same family as every other vacuous
gate on record: the check that cannot fail, next to output that looks like it passed.

## 0.25.1

### The workbench can now tell you whether its own answer is any good

Every number the browser workbench showed was a property of the state on screen — energy,
magnetization, sweeps, joules. None of them said whether a better state existed, which is the
question anyone running a sampler actually has. The C ABI gained the bounds in 0.25.0; the page now
uses them.

A **Solve** control runs tabu search, population annealing, or branch and bound, each reporting the
diagnostic only it can report: iterations actually run, `rho` against the population size, and
nodes-with-or-without-a-proof. A **Certify** button draws an **optimality bracket** — a bar from the
certified lower bound to zero, with the window where the true minimum still might be shaded, and a
tick at the state on screen. On the frustrated 12-ring preset: branch and bound proves −10 in 134
nodes, and the SDP bound independently says nothing below −11.591 exists.

**Two certificates, kept apart.** A gap of zero means the bound met the state, so no better state
can exist — proof by relaxation. `PROVED OPTIMAL` from branch and bound means the tree was
exhausted — proof by enumeration. The second can arrive while the gap is still wide, because the
tree closes on bounds computed per subproblem rather than the one on screen. The panel says which
argument it has, and the note explains the difference rather than blurring them into "optimal".

**The bounds are cached per graph and the bracket re-renders every frame**, so the window visibly
closes while an anneal runs, at no cost per sweep. That split is also the correctness boundary: a
bound is a property of the *graph* and survives a sweep; a proof is about the *state* and does not.
One sweep retracts the proof and keeps the bounds, and the browser tests assert exactly that —
along with the case that matters more, that loading a new model clears the old certificate. A
verdict that outlives its subject is the failure this panel exists to prevent, and it is a failure
that still *renders*.

### The committed wasm was seven days and one release stale

`docs/ferrotherm.wasm` was built on 20 August, before `tabu`, `popanneal`, `branch`, `sdp`, `host`
and twelve C ABI symbols existed. Rebuilt: 437 KB raw, 156 KB gzipped, 89 exports checked against
what the pages actually call. `check-wasm-exports.sh` caught the README's size figures in the same
pass — *a size a human retypes is a size that rots*.

## 0.25.0

### The optimality-gap certificate was reachable from one of six surfaces

`bound` is the headline claim in this crate's README — *a sampler holding a state of energy E is
within E − L of optimal whatever it found; at gap zero the answer is proven optimal without
trusting the sampler.* It had never been on the C ABI. Python, Julia, Zig, the HTTP server and the
MCP tools could build a graph and sample it, and could not ask how far from optimal the sample was.

`scripts/check-parity.sh` exists to catch exactly this — a capability that ships in Rust while the
other surfaces stay green because a missing binding is not a build error anywhere. It did not catch
it, and the reason is worth writing down: **it checks that every exported symbol reaches every
binding. A capability that was never exported is not a parity failure — it is a thing nobody can
say.** The gate protected the boundary it was pointed at, and the gap was one step upstream of it.

Twelve symbols close it, on the header, Python, Zig and Julia:

```text
  ft_tabu               ft_popanneal          ft_branch            ft_bound_decoupled
  ft_tabu_iterations    ft_popanneal_ln_z     ft_branch_proved     ft_bound_forest
                        ft_popanneal_rho      ft_branch_nodes      ft_bound_odd_cycle
                                                                   ft_bound_sdp
```

Three design choices in that list:

**Each solver leaves its best state as the simulation's state.** So `ft_spins` reads the answer and
`ft_energy` recomputes the energy from it rather than trusting the number the solver returned — and
they compose: anneal, then tabu from where annealing stopped, then branch and bound with that as its
incumbent. Every binding's test asserts the returned number matches the state left behind, because
those are two different claims and only one of them is checkable.

**`ft_bound_sdp` re-verifies the certificate before the number crosses.** It rebuilds the cost
matrix from the graph, re-runs the positive-definiteness proof, and returns NaN if that fails. A
bound crossing a language boundary is precisely the case where the caller cannot check it
themselves.

**`ft_tabu_iterations` and `ft_branch_proved` exist because the alternative is a number that looks
complete.** A tabu run shorter than its budget was truncated; a branch-and-bound run that exhausted
its nodes did not prove anything. Both are invisible from outside without a field to ask.

The bindings gained `Bounds` / `BranchResult` / `PopulationRun` types with `best`, `proved` and
`rho`, and Python and Julia gained a one-line `gap()` — `energy − best bound` — which is the number
the README has been describing since `bound` landed and which no non-Rust user could compute.

### And two operations the server never had

The HTTP API and the MCP tools offered five operations: `sample`, `anneal`, `energy`, `verify`,
`solve`. `verify` is the one that checks an answer — it enumerates the exact Boltzmann distribution
and reports the sampler's distance from it, capped at 20 nodes because enumeration is `2ⁿ`. There
was nothing that could check an answer at any other size.

`bound` is that operation. It returns all four lower bounds and, when handed a state, `energy −
best`: an upper limit on what a better search could still win, and **zero exactly when the state is
proved optimal without trusting whatever produced it**. It answers a different question from
`verify` — that one is about the distribution, this one is about the optimum — and unlike `verify`
it has no size cap, except that the SDP is `O(n³)` dense and is refused above 2048 nodes rather
than attempted, since an unbounded Cholesky is a way to take the server down with a valid-looking
request.

`optimize` exposes the three solvers with each method's own diagnostic in the reply:
`iterations_run` for tabu, `rho` plus a plain-English `rho_reading` for population annealing, and
`proved_optimal` / `hit_limit` for branch and bound.

**The MCP parity test only ran in one direction.** It checked that every advertised tool was
dispatchable — a tool with no operation behind it. It could not catch an operation with no tool in
front of it, which is the direction this project keeps failing in: `bound` and `optimize` were
dispatchable before they were advertised and nothing would have said so. The reverse check now runs
against the HTTP capabilities list rather than a constant. While adding it, a neighbouring test
that pinned the tool count at exactly 6 turned out to fail whenever a tool was ADDED — a failure
that reads as "the new tool is broken" and is fixed by editing a number, which teaches nothing. It
is a floor now, and completeness is checked against the API.

### Two things the new tests found

**A ferromagnet is proved optimal in under 300 nodes, and that is a fact about the instance.** The
budget-exhaustion test was written against a 64-spin Z1 grid on the assumption that 200 nodes could
not exhaust it. They could: a ferromagnet is unfrustrated, every coupling and every field is
satisfiable at once, so `−Σ|h| − Σ|J|` is not a relaxation there — it is the ground energy. The root
bound equals the incumbent and the entire tree prunes. Kept as its own test rather than discarded,
because it is the case where the cheapest bound in the crate is also the best available, which is
easy to forget after measuring bounds on G-set.

**Julia is 1-based and said so.** The Julia test was written with the 0-based indices the C ABI,
Python and Zig use, and `couple!` refused it: *`i = 0` is out of range for a model of 12 nodes;
indices here are 1-based.* An off-by-one that silently built a different graph would have been
scored with full confidence by everything downstream.

### `examples/exact_reach`: how far the exact solver actually goes

`exact_bracket` proves optima at 22 spins because that size is chosen to **always** prove — the tree
is smaller than the node budget, so the gate cannot silently stop applying. That makes it useless
for the question a user actually has, which is how big an instance they can hand this thing.

`exact_reach` walks the size up on a sparse and a dense family until the budget runs out, and
reports where it stopped. Not run in CI: it deliberately runs until it fails, which is the opposite
of what a per-push check should do. Measured, 40M-node budget, tabu incumbent, median of 3 seeds:

| family | mean degree | largest proved by 3/3 | nodes there | stops at |
|---|---|---|---|---|
| sparse | 6.0 | **76 spins** | 8,277,603 | 80 (1/3) |
| dense | 22.1 | **44 spins** | 12,173,789 | 48 (1/3) |

**Density costs far more than node count.** The bound charges for every edge with both ends still
free: a sparse graph has `O(n)` of those and a few fixings retire most of them, so the bound becomes
informative early; a dense one has `O(n²)`, so the root bound sits far below the optimum and stays
loose for many levels. 76 spins against 44 at less than a quarter of the node budget.

The prediction written into the example before it was run said the opposite — that a dense instance
would tighten *faster* per level, since fixing a high-degree spin retires many edges at once. That
is true per level and irrelevant to the outcome: there are quadratically more edges to retire.
Corrected in the file rather than removed, because the reasoning was sound and the conclusion was
not.

It also names where the next lever is. An SDP bound *inside* the tree rather than only at the root
is what the dense column is asking for, and that is what solvers of the BiqMac family do.

This measurement is why the timing column reads `spoiled` rather than being withheld: the reach and
the node counts are search outcomes, identical on any machine, and only the seconds are a rate. An
earlier draft called `require_quiet` and would have exited without printing any of it — a
misapplication of this workspace's own rule, on the same day the rule was written.

## 0.24.0

### A wall-clock number is a claim about the code only when the code had the machine

`examples/gset_gap` reported **85.7 s** for a G1 search that takes about 14 s on a quiet machine.
It reported it in the same format, with the same confidence, as every honest timing beside it —
because nothing in the example knew the difference. The machine's 1-minute load average was 189.

This is the **third** time this class has cost this project a number. The 86 ns/flip figure was
contaminated by background load and corrected. 0.17.0's `duty_cycle` table — 67.6 W idle, a 529%
idle-dominance threshold — was taken at a load average of 82, and `ferrotherm-meter` grew a guard:
it will not call a reading an idle baseline above a load average of 2, because **a busy machine has
no idle**. Both fixes were local to the file that had just been burned.

So the response this time was not a fourth local fix. The load-average reading moved down from
`meter` into the core crate — one implementation instead of two — and every example in the
repository was swept for the class, which found it in four.

The distinction the module is built around:

| | contaminated by load? | treatment |
|---|---|---|
| a cut, a bound, an energy, a state | no — same number whoever else is on the CPU | reported normally |
| flips/s, ns/flip, J/flip, a speedup column, a head-to-head | **yes** — every one is a division by wall-clock time | refused |

So `gset_gap` annotates and `flips_bench` / `parity_bench` stop:

```text
  cut found            11624   (8 restarts x 200-stage ladder, 85.7 s -- NOT A MEASUREMENT:
                                the 1-minute load average reached 189.4, so this is the run
                                queue's number, not the code's.)
```

`Timing::as_measurement` returns `Option<f64>`, so reaching the number means handling the
contaminated case. The defect was never a missing check — it was a check whose result nothing was
obliged to consult.

`FERROTHERM_ALLOW_BUSY=1` returns the numbers with a caveat printed above them, and CI sets it: a
hosted runner that has just finished `cargo build --release --examples` is exactly the machine the
guard is describing, and what that step checks is that the examples RUN.

### `sdp`: a certified SDP bound, and the mixing sweep made sparse

The max-cut SDP relaxation is the standard strong bound, and the obvious way to compute it produces
a *primal* value that bounds the SDP optimum from the wrong side — measured on G1, a naive
"upper bound" of 9579.94 against a max cut of 11624. Unsound by 2044.

`sdp` exhibits a **dual** point instead and checks one thing about it. For any dual-feasible `y`,
weak duality gives `eᵀy ≤ p* ≤ min_s E(s)` with no optimality, convergence, or rank assumption
anywhere — the mixing method, the rank, the Lanczos estimate and the seed are heuristics for
*choosing* `y`, and a bad choice moves the bound down rather than making it wrong.

The one claim that has to be true is that `C − Diag(y)` is positive definite, and that is
discharged by Rump's criterion rather than by an eigensolver: shift the diagonal down by a
computable constant and run a plain `f64` Cholesky, whose **completion proves definiteness**. No
`mul_add` anywhere in it — fusion changes the error model the theorem assumes, and a "harmless"
optimisation there would silently void the proof.

`Certificate::verify` rebuilds the cost matrix from the graph and re-runs the whole check, touching
nothing from the search that produced it. `gset_gap` calls it on every run. A bound that only its
own author can reproduce is not a bound.

The mixing sweep now reads a CSR copy of the cost matrix instead of scanning every column: on a
degree-4 instance at n = 800 the dense form made 800 column visits to find the 4 that mattered. The
CSR is built by scanning the dense rows in ascending column order, so every partial sum accumulates
in the same order — the change is **bit-identical**, and a test asserts that on the bit pattern
rather than on a tolerance. `assert!((a-b).abs() < 1e-12)` would pass just as happily for a
reordering that changed the arithmetic.

It buys time and **not tightness**, which is worth stating because the opposite was the working
assumption. Sweeping the sweep count says so outright:

```text
  sweeps      G11        G14         G1
     200      733       3409      12223
   1,000      731       3409      12224
   4,000      733       3409      12224
```

Twenty times the work, nothing to show for it. The mixing method reaches a stationary point early
and the bound is limited by something else entirely.

### The something else: `lanczos_min` was not returning an eigenvalue

`crate::linalg::jacobi_eig` **returns the eigenvector matrix** and leaves the eigenvalues on the
diagonal of the matrix it was handed. `lanczos_min` folded `min` over the return value. Eigenvector
components live in `[-1, 1]`, so the "estimate of `λ_min`" was the most negative eigenvector
*component* — a number near `-1` on every instance, with no relation to the spectrum of anything.

Nothing failed. That is the part worth keeping. The module's design claim is that Lanczos is *only*
a heuristic for choosing the shift — the Cholesky is what makes the bound sound, so a bad `θ` costs
tightness and cannot cost correctness — and the claim held exactly as written: every certificate
still verified, `gset_gap`'s independent re-check still passed, and the published-best-known
sanity check never fired. The bound was quietly loose on every instance instead of wrong on any.

It is also how the line survived. "Only a heuristic" is how a function gets no test, and a function
with no test does not stay a heuristic — it becomes a different function. There is a test now:
Lanczos is held against a dense Jacobi eigendecomposition of the same matrix, required to sit at or
above `λ_min` (Rayleigh–Ritz) and to reach it within `1e-3` from a full-length Krylov space.

**`examples/exact_bracket` is what exposed it**, on its first run, from a column nobody added for
that purpose. Across all six 22-spin instances the `sdp` column was *identical* to `decoupled` —
`certified` was falling back to its Gershgorin floor every time, because the shifted mixing point
never beat it. A gate written to check soundness found a tightness defect that no soundness test
could see.

What one line was worth, on the same six instances:

```text
  seed   optimum   decoupled   odd-cycle    sdp before    sdp after
     0   -13.872     -20.324     -19.119       -20.324      -14.934
     1   -13.633     -20.502     -17.524       -20.502      -14.541
     2   -18.623     -27.803     -24.420       -27.803      -19.952
     3   -20.618     -30.774     -27.638       -30.774      -21.625
     4   -20.499     -28.116     -24.804       -28.116      -20.865
     5   -14.740     -24.291     -22.564       -24.291      -15.849
```

The bound now closes a mean **88%** of the distance from `decoupled` to the proved optimum, having
closed none of it. And on G-set the corrected code lands on the numbers this module's own
documentation had been claiming all along — 629, 3192 and 12083, to the unit — which is the other
half of the story: the table was right and the code had stopped matching it, and nothing in the
repository was able to notice.

### `tabu`: the baseline a max-cut result is expected to be measured against

Tabu search is the mandatory comparison in the max-cut literature, and this crate did not have it.
The incremental move gain `Δ_i = 2 s_i (h_i + Σ_j J_ij s_j)` updates in `O(degree)` per flip.

It shipped **unsound** in its first form and the bug is worth recording: when every move was tabu
the loop `break`ed, so a run with a 50,000-iteration budget spent 9 of them and returned a result
indistinguishable from a completed one. On `n = 9` it missed the optimum on 7 of 30 seeds. The fix
is a restart rather than a return, a tenure clamped to `n − 1` so deadlock is unreachable, and
`Outcome::iterations_run`, because truncation was otherwise invisible from outside.

The control it is compared against changed too. `tenure: 1` had meant "no memory" only because of
an off-by-one; once that was fixed, the control became a second treatment. `steepest_descent` is a
real control algorithm instead.

### `popanneal`: an annealer that reports how much to believe it

Population annealing runs `R` chains down one ladder and resamples them at each rung. Two things
fall out that a single annealed chain cannot produce:

* **The partition function.** Each step's normalisation estimates `Z(β_k)/Z(β_{k−1})`, and the
  ladder telescopes into `ln Z`. Starting at `β = 0`, where `Z = 2ⁿ` exactly, makes it absolute
  rather than a ratio — and `Outcome::ln_z_is_absolute` is false when the ladder did not.
* **A diagnostic that can say "do not trust this run."** `ρ = (Σ_f n_f²)/R` over ancestor families
  is exactly 1 when every ancestor still has one descendant and exactly `R` when the population has
  collapsed onto one. A run whose `ρ` spiked explored one basin with `R` copies of one history.

Overflow is not a detail here. The reweighting factor is `exp(−Δβ·E)`, and G1's energies are near
`−2·10⁴`: a ladder step of `Δβ = 0.03` asks for `exp(600)` and `f64` overflows at `exp(709.78)`.
Unshifted, `Q` becomes `inf`, every `τ` becomes `NaN` and the population dies silently. Every
exponential is shifted by the running maximum, which cancels exactly in the ratios. The test for it
asserts the ladder **ran to the end**, not merely that `ln Z` was finite — a finite `ln_z` alone
would pass on a run that gave up at step one.

The flat-landscape test is exact rather than approximate: with no edges and no fields every
reweighting factor is exactly 1, so `Σexp = R`, `ln(R/R) = 0`, `τ_i = 1`, no Bernoulli is drawn and
no family is ever duplicated. `ln Z = n ln 2`, `ρ = 1` and the population size are all asserted with
`assert_eq!`.

### `branch`: the only thing here that returns a proof

Every other solver hands back a state, and `bound` hands back the other side of a bracket. `branch`
closes it: fix a spin, bound what the rest can still reach, discard the branch when the bound cannot
beat what is in hand. The bound is `decoupled` on the residual problem, maintained in `O(degree)`
per node rather than recomputed in `O(edges)`.

Undo is where this kind of search goes quietly wrong. `x + d − d` is not `x`, so an undo done by
arithmetic lets the bound drift as the search backtracks — and a bound that drifts **upward** prunes
a subtree containing the optimum while still reporting `proved_optimal`. Nothing is undone by
arithmetic here: scalars are restored by returning from the frame, touched entries of `λ` are saved
and written back verbatim, and the prune test carries an explicit slack sized from the instance for
what accumulates along a single root-to-node path.

`proved_optimal` is true only when the tree was exhausted inside the node budget. A run that hit the
limit says so, because a flag meaning "optimal, or else we gave up" gets read as the first thing and
quoted as the second.

### G-set, after all of it

```text
  instance  degree   cut found   best known     forest   odd-cycle     sdp      UB    gap
  G11          4.0         564   564 (100%)        817        *579     629    *579   2.6%
  G14         11.7        3058  3064 (99.8%)      4694        3602   *3192   *3192   4.2%
  G1          47.9       11624  11624 (100%)     19176       14958  *12083  *12083   3.8%
```

Starred is the bound that won; all three are sound, so the harness takes the maximum. G1's gap goes
from **22.3% to 3.8%** and G14's from **15.1% to 4.2%**, and G11 stays at 2.6% because on a degree-4
torus the cycle bound is still the strongest thing here — a semidefinite relaxation wins by more the
denser the instance, and a decomposition bound loses by more. G11's optimum is provably in
**[564, 579]**.

Timings are omitted from the table on purpose. Every number in it is a search outcome or a bound, so
it is the same on any machine; the seconds this run took are not, and the harness says so itself.

### CI has been failing since 0.23.0, and 0.23.0 was reported as green

`examples/gset_gap` takes a G-set file path and exits 2 without one. CI runs every unattended
example and requires exit 0. So from the moment that example landed, the example gate failed on
every push — and the release it landed in was reported as passing. The correction is stated here
rather than quietly fixed: `gh run list` says `failure` for the 0.23.0 commit.

Adding `gset_gap` to the skip list would have silenced the symptom and left the example unexercised,
which is the same outcome with better manners. It generates its own instance instead: with no
argument it emits an 8×8 ±1 torus — the shape of G11–G13 — **in G-set's own text format** and parses
it back through `Instance::parse`, so a bare `cargo run --example gset_gap` exercises the parser,
both decomposition bounds, the SDP, and the certificate's independent re-verification. No 200 KB
data file ships inside a source crate, and the example says outright that the built-in instance has
no published best-known cut and is comparable to nothing.

It also gained the check that needs no published number, and therefore runs on every instance:
**a cut this run actually achieved cannot exceed its own upper bound.** The existing gate compared
against the published best-known figure, which is only available for the real G-set files; this one
holds anywhere, and it exits 1.

### `cargo doc` is a published surface and nothing gated it

Turning `RUSTDOCFLAGS='-D warnings' cargo doc` on found **22 broken intra-doc links** already in the
tree. Most are prose that rustdoc read as a link and could not resolve — `[0,1]`, `reals[x]`,
`E[Delta]`, `[DS]` — and two are real: `[Verdict::Runnable]` and `[crate::graph::GRAPH_BUILDS]` name
items that do not exist (`Verdict::is_runnable` and `graph::graph_builds` do). All 22 are fixed in
the same commit that adds the gate, because a gate turned on over a failing tree is a gate that gets
turned off again.

### `examples/exact_bracket`: the bounds are now checked against ground truth on every push

Every bound in this crate was checked either against enumeration on graphs small enough to walk
(≤ 20 spins) or against a published best-known cut — which is itself only a lower bound and cannot
say whether a bound above it is wrong. Neither reaches an interesting size.

`branch` does. At 22 spins — a 4-million-state space, 256× what a unit test's `brute_min` can
enumerate — it returns the true minimum **with a proof**, so `decoupled`, `odd_cycle` and `sdp` can
be held against ground truth on six independent instances every push. The check is one-sided on
purpose: a lower bound may be loose by any amount and may never exceed the true minimum. The
heuristics are checked from the other side — `tabu` and `popanneal` may not report an energy *below*
a proved minimum, which would mean one of them is scoring states with a different energy function.

A run that hits the node budget exits non-zero rather than reporting what it found. Without the
proof the example checks nothing, and an example that silently degrades into checking nothing is the
failure mode the whole file exists to prevent.

## 0.23.0

### G-set, reported as a gap instead of a league-table entry

G-set has been the max-cut comparison set for twenty-five years, and every published figure is a
**best cut found** — a lower bound, which ranks how hard people looked and can never say whether
the best entry is optimal. `bound` supplies the other side, so the optimum is bracketed.

Measured, 800-node instances, 8 restarts, seconds each:

| instance | mean degree | cut found | best known | | upper bound | gap |
|---|---|---|---|---|---|---|
| G11 | 4.0 | 564 | 564 | **100.00%** | 579 | **2.6%** |
| G14 | 11.7 | 3058 | 3064 | 99.80% | 3602 | 15.1% |
| G1 | 47.9 | 11624 | 11624 | **100.00%** | 14958 | 22.3% |

We match the world best-known cut on G1 and G11. On G11 the true optimum is provably in
**[564, 579]**.

### `bound::forest` is worth nothing on max-cut, and now says so

A forest is **never frustrated**: any tree two-colours, so every edge in it is satisfiable at once
and the part's minimum is exactly `-Σ|J|`. Sum across parts and `forest` **is** `decoupled` whenever
the graph has no fields for the subgradient to move — which is every G-set instance. Measured:

```text
  G11:  decoupled -1600   forest -1600   -Σ|w| -1600
  G14:  decoupled -4694   forest -4694   -Σ|w| -4694
```

with `best_round = 0`, forty rounds of tightening improving nothing. **Trees cannot see the only
thing that makes max-cut hard.** Documented in the module rather than left for a reader to discover
from a suspiciously round number.

`bound::odd_cycle` charges `2·min|J|` for each **edge-disjoint** frustrated cycle — a cycle whose
coupling signs multiply to negative has at least one violated edge, and that term flips from `-|J|`
to `+|J|`. Edge-disjointness is what makes the penalties add: sharing an edge would double-count
the single violation paying for both. It takes G11's upper bound from 817 to 579 in 0.0 s.

Both are sound, so `gset_gap` reports the better of the two.

### `gset` exists to get one sign right

Max-cut is an Ising minimisation with the couplings **negated**. Load `J = +w` and
`cut = (W + E)/2`, so minimising energy *minimises* the cut: a plausible number, a valid state, and
the opposite problem — the kind of error that yields a complete benchmark table nobody can tell is
wrong. With `J = -w`, `cut = (W - E)/2` and a lower bound on energy becomes an upper bound on the
cut. A test proves the energy minimum coincides with the cut maximum by enumeration.

Two refusals aimed at harness failures rather than parser hygiene: a truncated download parses into
a valid **smaller** instance whose cut is incomparable with everyone else's (refused by edge count),
and a `0` in a 1-based file silently shifts every edge onto the wrong vertices (refused by range).

### A prediction, half right, and the wrong half recorded

`gset_gap` pre-registered that the bound would loosen with density. The trend held exactly — 2.6% at
mean degree 4, 15.1% at 11.7, 22.3% at 47.9 — but the prediction named the *forest* bound, which
contributes nothing here. Right about the phenomenon, wrong about the mechanism, and the example
says so where the prediction is.


## ferrotherm-gpu 0.3.4

### `cargo test` segfaulted on NVIDIA, and only a second vendor could have shown it

The WGSL sampler had run on Apple Metal, an NVIDIA L4 under Vulkan, and DX12 under WARP — a
software rasteriser. `GpuDevice` and the three defects `conform` found in it at 0.19.0 had been
verified on Metal alone.

On an RTX 4050, `cargo test -p ferrotherm-gpu` **SIGSEGVs**. Not a failure, not a skip: the test
binary dies after the first test that touches an adapter. Single-threaded it passes 12/12, so the
shader and the physics were never implicated — parallel Vulkan device creation and teardown crashes
the driver stack on that machine, which has `nvidia_icd.json` and `nouveau_icd.json` both registered
for the same physical device.

Environmental in origin and ours in effect, because a user running the default `cargo test` on a
common configuration got a crash. Both test modules now serialise adapter acquisition behind the
same std-only lock `ferrotherm-meter` uses for a structurally identical reason.

Two corrections on the way to it, both worth keeping:

- The guard must be **returned to the caller**. Bound inside the macro it drops at the end of the
  macro's own block and locks nothing for the test body — which looks exactly like a fix and is not.
- The expansion must be a **block expression** (`{{ }}`), or `let (gpu, _own) = gpu_or_skip!();`
  does not parse.

Verified on the configuration that crashed: 12/12 under default parallelism on the RTX 4050,
`conform` scoring the GPU path among them. "Runs on any fabric" now means more than one vendor's
silicon.


## 0.22.0

### The measurement, finally taken — and the instrument caught itself twice doing it

Blocked all release cycle for want of an idle machine. Taken on an x86 box at load 0.4 with Intel
RAPL, which reads **energy** directly rather than sampling power, so a window is a subtraction
instead of an integral estimated from samples.

Measured, i9-13900H, 1024×1024, 200 sweeps, one task of 209,715,200 node updates:

| cadence | duty | above idle | true total | understated |
|---|---|---|---|---|
| continuous | 100% | 41.4 J | 43.7 J | 1× |
| once a minute | 0.86% | 41.4 J | 309.0 J | **7×** |
| once an hour | 0.014% | 41.4 J | 16,095 J | **389×** |

Idle 4.5 W against 80.5 W marginal, so idle is most of the bill below a **5.5%** duty cycle, and
the standby budget settles at **4.47 W**. Extropic's `<1 W` Z1 spec clears that — a real margin, and
**~4.5×** rather than the 20× an assumed 20 W incumbent gives. Our own demo was flattering itself;
the measured incumbent is far more frugal than the modelled one.

### `psys` is a readable counter that reports computation is free

RAPL exposes `psys`, documented as whole-platform and exactly what an idle-draw argument wants. On
this part it is **dead**, and it fails in the worst possible way — readable, monotonic, plausible
units, disconnected from the machine:

```text
           IDLE      20 CORES BUSY
  psys     0.207 W      0.200 W     <- does not move
  package  3.324 W     75.973 W
```

`rapl::choose` checks the one invariant that cannot be argued with — a platform cannot draw less
than the chip inside it — and rejects `psys` when it reads below `package`. It would have supported
our own thesis, which is the dangerous direction.

### A meter and a device have to be in the same frame

The first run reported the GPU arm at 5.5 W marginal and 1.06 J for the task. The RTX 4050 is
**discrete**; RAPL `package-0` reads the CPU package; the card's draw is outside the counter
entirely. That number was the cost of *feeding* the GPU. On an Apple SoC the identical code is
correct, because `sys_power` covers the integrated GPU — so the error appears only when the backend
changes underneath it.

`Meter::scope()` returns `Scope::{WholeSystem, CpuPackage}` and `Scope::covers(discrete)` refuses
rather than divides. `duty_cycle` now skips that arm with the reason printed, and reports the path
it could see instead of discarding both.

Package scope also **understates** the machine — no RAM, storage, fans or supply losses — which
shrinks the incumbent's idle, the term the low-duty argument leans on. Conservative, and stated
wherever the numbers appear.

### Also

- `Meter::detect()` now routes macmon → rapl → ina3221. Every backend must be added there in the
  same commit: a backend nothing routes to is a backend nobody runs, which `ina3221` already
  demonstrated once.
- RAPL wraparound (262 kJ, about an hour at 75 W) is handled and tested; a wrap otherwise lands as
  one absurd negative sample mid-run.
- Subdomains are excluded by colon count: `intel-rapl:0:0` is the `core` subdomain of
  `intel-rapl:0` and double-counts its parent.


## 0.21.0

### A fresh survey, and a claim of ours that did not survive it

0.20.0 shipped saying optimality-gap certificates were a lane every stack leaves empty, reporting
only "best known". **That is false.** `dwave-preprocessing` exposes
`dwave.preprocessing.lower_bounds.roof_duality()`, which returns a lower bound on a binary quadratic
model's energy together with strong/weak-persistency variable fixings, and has for years. The error
was in our reading rather than their library: the survey that missed it lists `dwave-preprocessing
0.6.10` in its own D-Wave component inventory.

Withdrawn in `bound`'s module docs, the README and `docs/LANDSCAPE.md`. What remains, stated
narrowly: a lower bound in a std-only Rust stack, by a **different** relaxation — Lagrangian forest
decomposition rather than roof duality's max-flow construction — and an **anytime** one, since every
subgradient round is a valid bound. Which is tighter on which instances is **unmeasured**. Both are
sound, so their maximum is sound too, and that is what a caller with access to both should use.

### The standby figure is still unpublished — but active power is an upper bound on it

The core finding survived the survey intact: no vendor states what any of this hardware draws
between periods. What several do state is whole-device power **while working**. Extropic's Z1 spec
says `<1 W` sampling above 50 MHz — a projection for taped-out silicon, and absent from their
hardware page entirely. Fujitsu's Digital Annealer wants >100 W. A superparamagnetic-MTJ annealer
reports 0.64 mW. Normal Computing still publishes no watts at all beside "up to 1000×".

Active power bounds standby from above, because CMOS active power is leakage plus switching and
standby is leakage alone. So `DeviceRun::with_standby_at_most` takes the published active figure and
prices the device at or above what it really costs — which turns `StandbyUnpublished` from a dead
end into an inequality that can still decide the question.

The result is **asymmetric**, and `Verdict::outcome` enforces it rather than describing it:

- a device that wins while charged its full active draw wins with the real figure too →
  `ChallengerWins`
- a device that loses under that handicap has proven nothing, because the handicap may be the whole
  margin → `Inconclusive`, never `IncumbentWins`

On the worked example — Z1's `<1 W` against a machine idling at 20 W, one task a minute — the
outcome is `ChallengerWins`, from published numbers on both sides, for the first time.

### Also

- `DeviceRun` gains `standby_is_upper_bound` and `Verdict` gains `standby_was_bounded`, so a verdict
  carries the provenance of the number that decided it. Breaking for anyone constructing either
  literally.
- Confirmed unchanged by the survey: Normal Computing publishes no watts; no vendor publishes idle,
  standby, static or leakage power for any Ising machine.


## 0.20.0

### How far from optimal is that answer? Nobody in this field can say

Every sampler here returns the best state it happened to find, and none can say whether that is the
optimum. On anything too large for `exact`, nobody else can either: the field reports **"best
known"**, which is a statement about who has looked rather than about the problem. `planted` closes
this by constructing instances whose optimum is known in advance — and that only works on instances
you built.

`bound` closes it for instances you did not build. Split the energy into parts and minimise each
one independently:

```text
min_s E(s)  =  min_s Σ_k E_k(s)  >=  Σ_k min_s E_k(s)
```

The inequality is the whole method: the parts may disagree about `s`, so their separate minima can
only be lower than any single state's total. It holds for **any** split, which is what makes it safe
to optimise the split without ever risking an invalid bound. `bound::forest` splits the couplings
into forests — where elimination runs at induced width 1, exact and linear — shares the fields
across parts, and tightens by subgradient ascent until the parts stop disagreeing. That is
Lagrangian dual decomposition, and the output is an **optimality-gap certificate**: a sampler
holding a state of energy `E` is within `E - L` of optimal, whatever it found and however it found
it. At gap zero the answer is *proven* optimal, checkable without trusting the sampler.

Soundness is the only property that matters, because a bound above the optimum does not report a
small gap — it reports a **negative** one, and every conclusion drawn from it is backwards. Checked
against brute force on 200 random instances rather than argued.

### The test that could not see the thing its doc comment claimed

`forest` returns the best round, not the last, because subgradient ascent is not monotone. That
sentence sat in a doc comment and **nothing could check it**: mutating `if total > best` to take the
last round passed the entire suite, since `forest(g, r)` already maximises over rounds `0..r` and no
test could see inside it.

Measured instead of assumed: across 200 random instances at 40 rounds, **145 peak before the last
round**, so taking the last would be worse roughly three times in four. `Bound::best_round` reports
which round produced the value, which makes the difference observable from outside — and the
mutation now dies. Both ways this can silently stop being a bound are recorded rows.

### Also

- One test had its premise wrong: it asserted the forest split beats the decoupled floor on a
  *ferromagnetic* lattice, where every bond and field is satisfiable at once, so the floor is
  already the exact optimum and there is no room above it. The question only means anything where
  the floor is loose. It now checks both: strictly better on a frustrated instance, exactly equal
  on the unfrustrated one.
- `mutation-suite.sh` rows must be SINGLE-LINE patterns. A first draft embedded `\n`, which bash
  does not expand and the replacement never matched — the row would have reported MUTATION DID NOT
  APPLY, which the suite counts as a failure rather than a pass, so this one would have been caught
  loudly rather than quietly. Recorded anyway, because the next one might not be.


## 0.19.0

### The fastest sampler in the stack was the one path conformance could not reach

`ferrotherm`'s own survey has carried this line since 0.9.0: *"no `impl Device` for a GPU, so
`conform` cannot even score the GPU path."* It stayed true through five releases. The GPU sampler
could be **run** and could not be **checked against the fabric it claims to be**.

`GpuDevice` implements `Device`. Pointing `conform::run` at it took one line and found three
defects that being unscoreable had hidden:

**It returned the wrong state.** `Device::run`'s doc says "the final state" and every
implementation returns the BEST seen — `Cpu` delegates to `tempering::anneal_scheduled`, which
tracks the minimum over every sweep. The GPU returned whatever the coldest stage happened to stop
on, scoring **-57 against variable elimination's exact -59** on a ladder the CPU solves exactly.

**It could not have honoured a seed.** `Gpu::sweep` took none, so a `Device` matching the trait
signature would have accepted a seed and dropped it. That failure is invisible to every determinism
check ever written, because an ignored seed is perfectly reproducible: the caller varies it, gets
one answer, and reads a deaf sampler as a confident one. The shader's RNG is `hash(step, node,
const)`, so `sweep_seeded` offsets the dispatch counter by a mixing of the seed — no shader change,
and seed 0 maps to offset 0 so every Onsager number taken on this stream is exactly where it was.

**A run inherited the previous run's answer.** State was carried between calls and started at
all-minus-one, where `Cpu` builds a fresh `Sampler::new(g, beta, seed)` whose initial configuration
is drawn from the seed. So a second `run` began at the first one's best, could not improve on it,
and returned it — two different seeds, one answer. Matching `Sampler::new` exactly also means CPU
and GPU now start a given seed at the same configuration, which is what makes the two comparable.

### The f32 that was never declared

`GpuModel::{w,h}` are `Vec<f32>`: every coupling and field is rounded to 24 mantissa bits going in,
while the CPU path keeps f64. Every other fabric here declares its precision — D-Wave `Unstated`,
the fixed-point fabric `Fixed { bits }` — and the GPU declared nothing, so nothing downstream could
tell an arithmetic difference from a sampler difference. It is `Precision::Float { mantissa: 24 }`
now, and `Prices::UNSTATED` with the reason named: a GPU vendor publishes board power, which is a
rate for a whole card, not joules per spin update.

### Also

- `scripts/mutation-suite.sh` gains the seed-swallowing mutation, on the package field added in
  0.17.0 — without it the row would have compiled nothing and reported NO TEST MATCHED.
- Still open, and still in the landscape: the GPU is not reachable from Python, Julia, Zig, HTTP or
  MCP; there is no tabu search, branch-and-bound, dual/LP/SDP bound, or population annealing.


## 0.18.0

### The missing number is now a refusal, not a default

0.17.0 shipped `Machine::beaten_by`, which took the challenger's standby power and compute energy
as two bare `f64`s. That is usable and it is too easy to use wrongly: a caller can pass a standby
of zero and complete the comparison without ever noticing that **no vendor publishes one**, which
is precisely the assumption the module was written to expose. An API that lets the load-bearing
unknown default to zero has not exposed anything.

`DeviceRun` carries a device model at a cadence — its `Prices`, the `Ledger` of what it does in one
period, its graph size, and `standby_watts: Option<f64>` where `None` means *nobody has published
it*. `Machine::beaten_by_device` routes the comparison through it, and the checks arrive in the
order that makes a verdict mean something:

1. **Feasibility.** A fabric that cannot reflash inside the period loses at any price, and that
   verdict must not wait behind a number nobody has. `DeviceCannotSustain`.
2. **Is the computation priced at all.** `PricesUnstated`, distinct from a device that is
   expensive — `Prices::UNSTATED` is a fact about the literature, not the hardware.
3. **Standby.** `StandbyUnpublished`.

The headline test is `the_real_published_device_model_cannot_be_compared_at_all`: take Z1_SPICE,
the most completely specified device model in this field, run it model-resident so the reflash cap
is not what stops us, at a cadence it comfortably sustains. Its computation prices out at 1.77e-8 J
per period. The comparison still terminates, on one absent row.

That refusal names one number and nothing else, so it is a request rather than an objection —
`supplying_the_missing_number_makes_the_arithmetic_run` supplies a hypothetical milliwatt and gets
a verdict identical to the loose-number form.

### `duty_cycle` now says something true on a machine it cannot measure

The example needed a power meter and a quiet machine, so on a loaded one it printed two refusals
and stopped — correct, and useless. The device-side half needs neither. It now runs first and
unconditionally, prices the vendor's own model, and prints exactly where the comparison stops.
An example whose whole output is "I could not run" teaches nothing about the thing it exists to
show.

Breaking: `DutyError` gains three variants, so exhaustive matches on it need updating.


## 0.17.0

### Every energy number this field publishes leaves out where the joules actually are

`joules.rs` measures both paths **above idle** and divides by work done. So does every energy
comparison this project has published, and so does every one the rest of the field has. It is the
right question for a machine kept busy, and it quietly assumes the thing most workloads do not do.

A sensor drawing from a posterior ten times a second computes for microseconds and waits out the
rest of the second. Subtracting idle prices the microseconds and throws away the wait — and the wait
is where the joules are. The new `duty` module prices the wait:

```text
E over one period  =  marginal × t_run  +  idle × period
```

### The standby budget: one number that decides the whole value proposition

Inverting the same arithmetic gives what a competing device has to hit:

```text
standby budget = idle + marginal × duty
```

It grants the challenger **perfectly free computation**, so a better sampler cannot argue it down —
whatever it rules out stays ruled out. And as the cadence slackens it collapses onto the incumbent's
idle draw, with nothing about sampling left in it.

Which is where the field runs out of numbers. `ledger::Prices` states `e_sample`, `e_read`,
`e_write` and a reflash cap because Table IV states them. It carries **no standby term, because no
thermodynamic vendor publishes one** — so the device column of that comparison cannot be filled in
by anyone today, and the strongest available argument for a sampling fabric (the intermittent,
low-duty edge workload) is the one nobody has priced.

`duty` refuses a cadence the machine cannot sustain rather than pricing a run that could not have
happened — the same refusal `Ledger::reflash_seconds` already makes on the device side. Seven tests,
none of which need a power meter.

### A busy machine has no idle, and `Meter` now says so

`Meter::idle` has always guarded the *delta* against the baseline's noise. It never checked that the
baseline was **idle**, and that hole has already cost this project a published number: the 86
ns/flip figure corrected in an earlier release was contaminated by concurrent background load.

It cost one again here. The first run of `duty_cycle` produced a complete, plausible table — 67.6 W
idle, a 529% idle-dominance threshold, an intermittent workload understated by five orders of
magnitude — on a machine at a 1-minute load average of **82**, with five other sessions running
experiments and Rust builds. Nothing looked wrong. The numbers were somebody else's work charged to
ours.

`Meter::idle` now reads the load average and refuses above 2 runnable threads. `Baseline` carries
`load1` so a figure published from it can be audited later, when the machine is no longer in that
state. The threshold is 2 rather than a fraction of the core count because a load average counts
runnable threads, not utilisation: two threads that never sleep are two cores' worth of heat whether
the machine has four cores or forty.

**Which way this biases matters.** Other people's work *inflates* a baseline, so idle looks larger
than it is — and the argument this release is built on turns on how much of the bill is idle. The
contaminated run flattered our own conclusion. Those numbers are withdrawn and the measurement is
reported as pending rather than published.

### `sweeps_par`'s throughput figure is not reproducible as stated

The README quoted 3.8e8 flips/s at 18 threads without naming the lattice, which turns out to be the
load-bearing detail: `sweeps_par` spawns its threads *inside* each sweep, so parallel efficiency is
set by how much work one sweep carries and the same call reports different speedups at different
problem sizes. Read off the source, not measured — re-measuring is pending a quiet machine.

### Also

- `duty_cycle` reports every path the instrument could see. An earlier version returned early unless
  **both** measured, which threw away a good GPU measurement because the CPU's delta sat in the
  noise — and the finding here is a property of one machine at a time, not of a ratio.
- The example states which comparison it is making: the meter reads whole-system power, so its idle
  term prices a fabric that **replaces** the host. A fabric sitting *beside* a host that stays awake
  anyway is a different sum, and an integrated SoC shares one rail, so per-component idle is not
  separable on this machine. Named rather than split by assumption.
- `Baseline` gains a public field, which is a breaking change for anyone constructing one literally.

## 0.12.0

### `x >= 3` returned a confident optimum to an unbounded problem

Found by clippy, of all things: `i64::MAX.min(num(lo)? + 1)` in the LP bound parser, flagged as an
expression with no effect. It reads like an overflow guard and is not one — the addition happens
first — and what it actually did was give every one-sided lower bound an **invented upper bound of
`lo + 1`**. So `t >= 10` compiled to the domain `10..=11`, and

```text
Maximize
  obj: t
Bounds
  t >= 10
General
  t
End
```

answered **11**. Not an error, not a warning: a confident optimum, for a problem that has none.

LP format's default upper bound is `+infinity` and there is no such spin domain, so this is now
refused by name with its line number, the way every other unrepresentable input already was. The
one-sided `x <= 5` form stays legal, because LP's default *lower* bound really is 0.

Every bound test used the two-sided `10 <= t <= 20` form, which is why nothing looked here for as
long as the parser has existed.

### `ferrotherm-cloud` and `ferrotherm-gpu` are on crates.io

`ferrotherm-gpu` had been at 0.2.0 in the tree and 0.1.0 on crates.io. `ferrotherm-cloud` — the
Hitachi CMOS annealing driver, 615 lines, 11 tests, API conventions established empirically rather
than read off a document — had never been published at all, and was named in no README. Both are out
now, both are in the README's crate table, and `RELEASING.md` writes down the publish order that
previously lived in nobody's head. See the Unreleased notes below for the gate that found them.

### Also

- The Hitachi fabric is in the field map, where it belonged from the day it was written: real
  fabricated Ising silicon, 384×384 King's graph, four-bit coefficients, over a free public API.
- A layout closure in `ferrotherm-cloud` was named `push`, pushed nothing, and took two coordinates
  it never used — so every reader had to check whether couplings were being written twice.
- Zero warnings across the workspace.

## 0.16.0

### Six cross-binding drifts, and the gate that catches the class

`Problem.solve` in the Zig binding never called `ft_model_compile`, so it could not solve anything —
and every gate was green. That is the finding worth keeping. `check-parity` proves the **names**
reach each binding; `check-semantics` proves each **builds** byte-identical `.ftp`. Neither could see
it, and check-semantics could not *by construction*: it compares the model before solving, and its
own harness calls `compile()` explicitly, so it exercised the very step `solve` was skipping.

`scripts/check-answers.sh` closes that: one model with a deliberately **unique** optimum, solved end
to end on five surfaces, same answer required. Its Zig harness deliberately does *not* call
`compile()`, because a gate that works around the defect it hunts is worth nothing.

An adversarial read of every binding against `include/ferrotherm.h` then found six more, each
confirmed by a second agent told to refute it (seven further claims were refuted and dropped):

| surface | drift |
|---|---|
| `serve` | `/v1/anneal` ran 0.1→3.0 over **480** sweeps; Python, Julia and the browser IDE all use 0.05→4.0 over **2400** |
| `serve` | `/v1/solve` defaulted to **16** tries; Python and Julia both use **12** |
| `zig` | a duplicate name was accepted and **silently renamed** to the synthetic `v1` |
| `zig` | `ommx()` before compile returned an **empty slice**, not an error |
| `zig` | `value()` collapsed "never solved" with "solved but undecoded" |
| C ABI | `not_equal`/`equal` returned 0 without setting `ft_model_error`, which the header promises |

Three surfaces agreeing and one differing is drift, not a design choice.

### Breaking

- Zig `Problem.value` returns `Error!?i64` — `null` is a solved-but-undecoded variable, matching
  Python's `None` and Julia's `nothing`.
- Zig `Problem.ommx` returns `Error![]const u8`.
- Zig `Problem.binary`/`categorical`/`integer` return `Error.DuplicateName` for a name already taken.
- `ferrotherm-serve` 0.8.0: `/v1/anneal` and `/v1/solve` defaults now match the library's.

## 0.15.1

### PyPI was serving the Rust README

`python/README.md` — the description PyPI shows — was a **verbatim copy of the Rust one**. Someone
who ran `pip install ferrotherm` was told to run `cargo add ferrotherm` and shown Rust code. It also
still carried the `joules` snippet that does not compile, because that fix was applied to the root
README and the copy was forgotten.

Rewritten as an actual Python README: the `Problem` API with named variables and named results, soft
constraints, direct sampling, and the ledger — using the real accessors, which are properties
(`s.magnetization`, `s.node_updates`, `s.joules`) and not the methods the first draft guessed at.

`python/test_readme.py` executes **every** ```python block, standalone, and holds each `print` line
to the value written beside it. Running them is what found `magnetization()`, and a `ledger()` that
does not exist.

The Onsager block used 16×16 over 500 sweeps and printed 0.898 beside Onsager's 0.974 as though they
agreed. A small lattice sampled briefly is not the thermodynamic limit. It is 64×64 over 2000 sweeps
now — four decimals, still instant — and says why the size is not decoration.

Two rounds of the test being wrong before it was right, both found by reverting the README and
watching it stay green: the first version checked **hardcoded** constants rather than the file, and
the second searched the whole of stdout for the claimed number, so a drifted `|M|` was satisfied by
the *next* line's output. It matches per line now.

## Unreleased

### `ferrotherm-silicon` 0.2.0 — the deprecated `pbit_*` aliases are gone

`pbit_threshold_init` and `pbit_fire_prob` were kept "so 0.1.0 callers keep building", with a test
pinning that promise. `ferrotherm-silicon` 0.1.0 has fifteen downloads, which is what crates.io's own
mirroring produces for a crate with no dependents. The shim was protecting nobody and the test was
spending CI on it.

Use `bsn_threshold_init` and `bsn_fire_prob` — the literature name, which is what they were renamed
to in the first place. This is a pre-1.0 project and a correct API beats a compatible one.

## 0.15.0

### The OMMX decoder, rewritten from the specification

OMMX has produced **six** defects — more than the rest of this crate combined. Shipped as a sibling
crate the C ABI could not reach. Julia functions exported with no body. The constant documented
backwards on five surfaces. proto3's default-omission rule missed, so variable 0 read as "not
declared". A length prefix that wrapped `usize` and crashed on eleven bytes. A diagonal quadratic
term — *valid* OMMX — that aborted the host on twenty-three.

That is not six unlucky mistakes. It is one structural fault with six symptoms, and it was in the
decoder's type:

```rust
fn next(&mut self) -> Option<(u32, Body)>          // the old one
fn read_field(&mut self) -> Result<Option<Field>, WireError>   // now
```

`None` meant **both** "the message ended" and "the message is corrupt". A truncated instance parsed
as a shorter valid instance, silently, so every corruption case had to be caught by hand somewhere
further up — which is precisely the whack-a-mole the six defects record. A decoder that cannot say
"malformed" makes every caller guess.

`src/wire.rs` is the replacement: the protobuf wire format written against the spec rather than
against the instances that happened to turn up. Beyond the type change it fixes four things the old
one had wrong:

- **`fixed32` returned zero.** The arm was `self.i += 4; Some((field, Body::Varint(0)))` — four bytes
  advanced with no bounds check and **the value replaced by 0**. Any 32-bit float field would have
  read as 0.0 with nothing said.
- **Groups (wire types 3/4) silently truncated the message.** Deprecated, not impossible; a reader
  that stops at one reports a partial message as whole. Refused by name.
- **Varints were not canonical.** At most ten bytes, and the tenth carries one bit. Longer is
  malformed input, not a large number.
- **Packed arrays truncated on error.** `while let Some(..)` returned the prefix it managed — the
  same fault as the message-level one, one level down.

Field number 0 is refused. Every length goes through `usize::try_from`, never `as usize` — the cast
is what wrapped, and a wrapped length passes any bounds check written after it.

**Validated against the format's own implementation throughout.** `tests/ommx_reference.rs` shells
out to Python's `ommx` package and scores every state; it passed before the rewrite and passes
after, so the wire change is provably behaviour-preserving where it matters.

The new fuzz target `the_wire_codec_never_panics_and_never_reports_a_truncation_as_an_end` states the
property directly: a cut landing **on** a field boundary is a legitimate shorter message, and a cut
landing anywhere else must be an error. That took two attempts — the first version asserted only
"a prefix never reports all four fields", which is true of the broken decoder too, since a short
message genuinely has fewer complete fields. Reverting the fix and watching it stay green is how
that was found.

### Breaking

- `ImportError` gained a `Wire(WireError)` variant, so an exhaustive match on it no longer is.
- `ommx::import` now returns an error for byte strings it used to accept as a shorter instance. That
  is the point of the release: those inputs were being silently misread.
- New public module `ferrotherm::wire`.

## 0.14.0

**Two unbounded allocations, found by a fuzzer written after 0.13.0 shipped.**

The 0.13.0 audit found seven crash paths by *reading code*, which is not a repeatable method. So the
parsers now have a fuzz harness — and it found two defects on its first two runs, both of a class the
reading had missed entirely: not a panic, but an **allocation sized directly from a number in the
input**.

- **Fifty bytes asked for 96 GB.** `spins` was unbounded, which made the colour-class bound
  `c < spins` **vacuous** — a declared spin count of `u64::MAX` admits every colour index there is.
  `spins 18446744073709551615` then `color 4000000000 0` allocated 4,000,000,001 colour classes.
  Two independent bounds now: `spins` is capped at `u32::MAX`, the graph's own index type, and
  colour classes at 2²⁰ — a graph needing a million colours has a vertex of degree 999,999.
- **A six-line LP file asked for 1 GB.** `lp::parse` emits one objective term per value of an
  integer's domain, so a range spanning most of `i64` is an unbounded allocating loop — and it runs
  *during parsing*, before `compile()` and therefore before 0.13.0's `DomainTooLarge` refusal could
  see it. Refused at the declaration now, with the same `u32::MAX` ceiling so the two agree.

### The harness

`tests/fuzz_parsers.rs` (ftp, LP, OMMX) and `serve/tests/fuzz_json.rs` (the parser with the proven
remote DoS). No fuzzing crate — the core keeps its zero dependencies — just a seeded xorshift, so
every failure reproduces exactly. Three input shapes: random bytes, mutated valid inputs, and the
specific integers that break arithmetic rather than parsing.

Both carry a **capped global allocator** that refuses any single allocation over 512 MB. That is not
decoration: a fuzzer's purpose is to find the input a parser mishandles, and when the mishandling is
an unbounded allocation, *finding* it means the machine tries to serve it. The first run of this
harness sent a laptop to swap. A refused allocation aborts instantly instead, and
`the_allocation_cap_actually_fires` checks in a child process that the cap is wired to something.

Verified by reverting both `ftp` bounds and watching the harness report
`memory allocation of 96000000024 bytes failed` — and worth recording that the *first* version of
the hostile-input list would **not** have re-found its own defect, because it had `spins u64::MAX`
and a large colour index as separate cases when only their combination allocates.

### Breaking

- `ftp` refuses a spin count above `u32::MAX`, and a colour class index above 2²⁰.
- `lp::parse` refuses a `General` variable whose domain exceeds `u32::MAX` values.

## 0.13.0

**A crash-safety release.** Seven ways a caller could abort the host process, one silently wrong LP
answer, an exact reference that returned NaN, a verification gate that switched itself off, and a
header that lied about its own arity — all found by an adversarial audit, each confirmed by a second
reviewer told to refute it, each fixed with a test that fails against the old code.

If you link `ferrotherm` from C, Python, Julia, Zig or a browser, **upgrade**: on 0.12.0 eleven bytes
of malformed OMMX, or 45 bytes of malformed `.ftp`, end the process that called you.

### Breaking

- `lp::parse` now refuses a `Bounds` line on a `Binary` variable instead of dropping it. Files that
  used to parse and return a wrong answer now return an error naming the variable.
- `Ebm::new` asserts its node count fits the `u16` edge index space.
- `CompileError` gained `NotFinite` and `DomainTooLarge`; a match on it that was exhaustive is not.
- `ising::tv` asserts its two distributions are the same length.
- `serve` bounds a request by compiled **couplings**, so a model that used to be served slowly is now
  refused quickly.

The unreleased notes below detail each.

## Unreleased

### Two ways to overload the server from a tiny request

- **`/v1/anneal` wrapped its own budget.** `(stages * per) as u64` multiplies in `usize` *before*
  the cast, so `"stages": 9223372036854775808` produced a small number, sailed past the node-update
  ceiling, and aborted the process in `raw_vec` with a capacity overflow — an empty reply, no 400,
  and the server gone. It saturates in `u64` throughout now, and the ladder is clamped the way
  `/v1/solve`'s already was.
- **`/v1/solve` measured the wrong dimension.** Its update bound sat *inside* the schedule arm, so a
  request naming no schedule had none at all. But the deeper problem was that no ceiling measured
  what actually grows: `{"variables":[{"name":"x","values":1000}],"tries":1}` is **46 bytes**, and
  compiles to only 1000 spins — far under `MAX_NODES` — while a one-hot over k values carries
  k(k−1)/2 **couplings**. That is 499,500 of them: 6.7 s and a **17 MB** reply.

  So there is a `MAX_COUPLINGS`, and its value is measured rather than guessed — cost is linear in
  couplings at ~34 bytes and ~13 µs each:

  | one-hot k | couplings | reply | wall |
  |---|---|---|---|
  | 100 | 4,950 | 161 KB | 0.08 s |
  | 300 | 44,850 | 1.5 MB | 0.58 s |
  | 600 | 179,700 | 6.2 MB | 2.38 s |
  | 1000 | 499,500 | 17 MB | 6.73 s |

  100,000 holds a request to ~3.4 MB and ~1.3 s and still admits a one-hot over 447 values. The
  refusal names the dimension that grew and points at `Encoding::DomainWall`, which is linear.


### Thirteen claims that were not true, including the first line anyone copies

The audit's last lens read every README and module doc against what the code actually does. The
headline is the smallest: **the README's "Use it" snippet did not compile.** `Ledger::joules` returns
`Option<f64>` and the snippet formatted it with `{:.2e}`, which `Option` does not implement — so the
first thing a new user pastes fails, and nothing was checking it. The README is now included as a
`#[cfg(doctest)]` carrier, so `cargo test` compiles it; reverting the fix makes that test fail with
the original `LowerExp` error.

Measured and corrected:

| claim | was | is |
|---|---|---|
| wasm size | 44 KB | **356 KB** (128 KB gzipped) |
| `cargo test` | "6/6" | **507 tests** across six crates |
| `compile_chain` readout KL | 0.0057 | 0.0054 |
| `compile_chain` ε comparison | 0.721 vs 0.754 | 0.721 vs 0.750 |
| `reach_on_z1` per-joint factorization | 35–45% | **32–35%** |
| examples that are gates | "every example" | **7 of 20** |

Three were not arithmetic:

- **"Every one of them reaches the hardware through the same `fabric::Device` trait"** — `-gpu`,
  `-meter` and `-serve` implement no `Device`. Only `-cloud` and `-silicon` do, which is now what it
  says.
- **`reach_on_z1` was not reproducible.** `HashMap` iteration order plus a sort with no tiebreak
  meant four runs gave three different answers, and the figure the README quoted appeared in none of
  them. It is `BTreeMap` with a total order on `Key` now: three runs, byte-identical output. An
  example cited as evidence has to be reproducible or it is not evidence.
- **`reach_on_z1` printed "+3 points" inside its own READING block** as though the run had measured
  it. Post-training here reports NLL and never evaluates a success-rate delta. It is labelled as a
  prior result now.

Also: two bare absence claims became "this review did not locate"; `serve/README`'s two TV figures
said they "are asserted in the test suite" when only their *direction* is; and `meter/README` still
said the INA3221 backend was "Not implemented" hours after it shipped — while `Meter::detect()`
genuinely could not reach it. `detect()` is `macmon().or_else(ina3221)` now, because a backend
nothing routes to is a backend nobody runs.

### Clippy runs in CI, and the workspace is clean at `-D warnings`

The tool that found the 0.12.0 headline defect ran in no job and no script. It now runs over the
whole workspace with warnings denied, and the count is **zero**. Three lints are allowed in
`[workspace.lints.clippy]`, each with its reason recorded there rather than as scattered
`#[allow]`s: `needless_range_loop` (the index is a spin, a site, a colour class), `too_many_arguments`
(a physics kernel takes the parameters the physics has), and `neg_cmp_op_on_partial_ord` — where
following the advice would have been a **bug**, since `!(w > 0.0)` rejects NaN and `w <= 0.0` accepts
it.

One test made honest rather than tolerant: the meter's real-workload test panicked when a concurrent
build contaminated its baseline. That is the meter's refusal working, not a code defect, so it skips
with the reason — the same treatment its `measure` call already had, applied to the `idle` line that
was missed.


### Four gates that were not gating

The same audit turned its lens on the checks themselves. Every one of these read, in a workflow, as
a step that passed.

- **CI never compiled three of the six crates.** The step was `cargo test --release`, which builds
  and tests only the root package — so `ferrotherm-serve`, `ferrotherm-cloud` and
  `ferrotherm-silicon` were never compiled by CI at all, and their **131 tests never ran**. It is
  `--workspace` now, and the run does compile all six.
- **`check-exports.sh` lived in a job with no Julia.** It printed "skipping" and exited 0 on every
  CI run since it was written, having checked nothing. It has moved to the `julia` job, and
  `FERROTHERM_REQUIRE_JULIA=1` — which CI sets — turns the skip into a refusal, so it cannot go
  quiet again. A local run without Julia still skips, which is right.
- **`check-wasm-exports.sh` matched substrings of the whole binary.** `grep -q "$sym" "$wasm"` is
  true if the name appears anywhere: inside a longer symbol, in a debug string, in a data segment.
  Demonstrated on the shipped artefact — the fake symbol `ft_ener` matches, because the real export
  `ft_energy` contains it. Names are now matched length-prefixed against the export section, which
  rejects `ft_ener` and still finds all 77 real exports.
- **`check-semantics.sh` called a compile failure a missing toolchain.** `cargo build -p
  ferrotherm-serve 2>/dev/null` falling through to `skip http/mcp "serve did not build"` reads as
  "this machine lacks something" and exits 0. The machine lacks nothing: the crate is in this
  repository and it is broken, and the compiler error was discarded on the way past. It exits 2 with
  the error now — verified by breaking `serve/src/lib.rs` on purpose.
- And the wasm binding's stderr was still being discarded — the one binding yesterday's pass missed,
  which is why a failing wasm printed "(it wrote nothing to stderr either)".


### Seven ways a caller could abort the host process, and one wrong answer

An adversarial audit of the parsers, the C ABI and the server found nine defects, each confirmed by
a second agent told to refute it. Seven of them **abort the calling process** rather than returning
an error — across a C ABI a panic is non-unwinding, so a bad byte in a file kills the program that
linked this library.

| defect | what it took |
|---|---|
| `ommx::import` length prefix wrapped `usize`, skipping the bounds check | **11 bytes** |
| An OMMX diagonal quadratic term (`row == col`) reached the builder as a self-edge | **23 bytes** of valid OMMX |
| `serve`'s JSON parser had no depth limit | **40 KB** POST killed the whole server |
| `ft_ising2d_new(1, ..)` built the self-edge `(0,0)` | one call |
| `ft_planted_wishart(3, inf, ..)` — the guard admitted `+inf` while refusing NaN | one call |
| `ft_model_integer` accepted a range spanning most of `i64` | two calls |
| `ftp::Program::from_ftp` trusted the colour index; `c + 1` overflowed | **45 bytes** |

The diagonal quadratic is the one worth dwelling on: `row == col` is **ordinary, well-formed OMMX**
that other tools emit routinely. For a binary `x`, `x² == x` exactly, so it is now folded into the
linear part rather than refused — refusing would have rejected valid input to stop a crash.

**And one silent wrong answer.** `lp::parse` read a `Bounds` line on a `Binary` variable into its map
and never consulted it, because the binary path does not look there. `x <= 0` fixes x **off**; the
model solved as though x were free and returned a confident answer to a different problem — from a
file the caller did not write and cannot check by eye.

**A header that lied about its own arity.** `include/ferrotherm.h` declared `ft_gpu_class_ptr(const
ft_sim *)` while the function takes `(sim, uint32_t c)`. C callers compiled **without a single
warning** under `-Wall -Wextra` and read whatever was in the second register: every colour class came
back `NULL`, with no diagnostic. Verified fixed from C — the classes now return real pointers.

### Also

- `scale_to_fit`'s repair needed one more line than the fix above: above 2^53 a decrement of 1.0 is a
  **no-op** in f64, measured on `fujitsu_da3` where the ceiling is 3.07e18. Without a `next == n`
  break the loop re-tests one candidate a million times. It now answers in 791 ns.


### The exact reference distribution returned NaN at the betas its own schedules use

`ising::exact_boltzmann` is the oracle every sampler in this crate is verified against. It
accumulated `(-beta * energy).exp()` directly, and `f64` overflows near `exp(709)` — so on an
8-spin ferromagnetic ring a large beta sent `z` to `+inf` and every probability to `0` or `NaN`.

Measured before the fix: a 4×4 lattice is clean at β=20 and returns 2 NaN entries at β=24; a 24-spin
complete ferromagnet, inside the documented `n ≤ 24` limit, is clean at β=2.5 and produces NaN at
β=3.0. **The crate's own schedules run to β=6 and β=8.** So the oracle stopped answering inside the
range it is used in, and `certify` swallowed it silently, because `NaN > floor` is false.

Fixed by subtracting the maximum log-weight before exponentiating — exact, since the shift cancels
in `w/z`. `HetSampler::sweep` already did this; the enumerations never got it. Applied to all three:
`ising::exact_boltzmann`, `het::exact_boltzmann`, and both accumulators in `Dtm::exact_log_cond`,
which ends in `num.ln() - den.ln()` and so wanted log space all along.

### `certify`'s distributional gate switched itself off as models got bigger

The sampling-noise floor is `0.5·√(2ⁿ/ess)`, which passes 1 as `n` grows — and total variation
between two distributions can never exceed 1, so `tv > floor` becomes **unsatisfiable**. The gate
went quiet on exactly the models it matters for, `Certificate::passed()` counted the silence as a
pass, and `noise_floor: Some(2.04)` was still printed as though it were a real threshold.

Measured with iid uniform noise, which has no relation to the model at all: at n=9 and 4000 draws it
is caught; at **n=16 and the same 4000 draws it is not**; at 20000 draws it is caught again. A
vacuous floor now reports `TooFewSamples` instead of nothing.

`AboveNoiseFloor` appeared in the enum, in `Display`, and at one push site — and in **no test, at any
n**. There is one now, and it fails against the old code.

### `jacobi_eig` was not scale invariant

Both convergence thresholds were absolute constants. A well-conditioned SPD matrix scaled down far
enough has every off-diagonal below the pivot threshold, so no rotation is ever applied and the
function returns the **untouched diagonal** as its eigenvalues with the identity as eigenvectors.
On `[[3,1,0.5],[1,3,1],[0.5,1,3]]`: correct at scale 1 and 1e-6, wrong in the second digit at 1e-11,
exactly the input diagonal at 1e-13. Downstream, `tla::solve_spd_exact_ou` checks only that the
eigenvalues are positive — which the bogus ones are — and returned a component with the **wrong
sign** and no error. The thresholds are relative to the Frobenius norm now, and the test runs the
same matrix across 26 orders of magnitude.

### Also

- `ising::tv` zipped two slices of different length and returned a **truncated** distance:
  `tv(&[0.25; 4], &[0.5, 0.5])` gave 0.25 where the honest answer over the shared 4-state space is
  0.5. Truncation always under-estimates, and every use of `tv` has the shape
  `assert!(tv < tolerance)` — the failure direction that turns a red test green. Refused now.
- `Ebm::new` took a `usize` node count with `u16` edge endpoints and no check. Past 65,536 a caller
  narrowing its own indices aliases them and builds a different graph **with no error**, since every
  truncated index is still `< n`. `pattern_grid(300, ..)` collapsed 90,000 nodes to 65,536 and
  introduced an odd cycle, silently degrading the chromatic sweep to sequential.
- `Ebm::is_bipartite` re-inferred its answer as `!classes[1].is_empty()` rather than returning what
  the BFS had already concluded, so an **edgeless graph reported false**.

All five were found by an adversarial audit and confirmed by a second agent that was told to refute
them; each fix carries a test that fails against the old code.


### A NaN objective coefficient silently disabled every other preference

The worst kind of bug this crate can have, and it compiled, solved, and reported success.

```rust
m.objective(Sense::Maximize, Expr::product(3.0, &[Lit::Is(a, 1)]));
m.objective(Sense::Maximize, Expr::product(f64::NAN, &[Lit::Is(b, 1)]));
```

`compile()` returned `Ok`. `solve_best_of(64)` returned `feasible: true`. And `a` came back **0**,
when it is worth +3. Every comparison against NaN is false, so the sampler's "is this better than
the best so far" test never fires again and the search silently stops improving. A confident,
feasible-looking, wrong answer.

Non-finite coefficients are now refused at `compile()` — the one place `objective`, `set_objective`,
the LP reader, the OMMX reader and the C ABI all converge, because checking in `objective` alone
would leave the other four open. The message distinguishes NaN from infinity, since they fail
differently, and notes that coefficients are stored as "minimise this" so a maximised term appears
negated.

Found while demonstrating that a *different* clippy lint should be allowed: `!(w > 0.0)` rejects NaN
where clippy's suggested `w <= 0.0` accepts it, and checking whether the library actually refused a
NaN weight showed that it did not.

The first cut of the fix rejected every **hard** constraint, because a hard constraint carries
`f64::NAN` as its sentinel. The FFI cardinality tests caught it immediately. The sentinel is
deliberate; the check now knows that, and the test asserts a hard constraint still compiles.

### `scale_to_fit` bounded the answer instead of the work

`Fabric::scale_to_fit` returned `None` — "scaling cannot help" — whenever the candidate ceiling
exceeded 1e6. The search walks candidates **downward from that ceiling**, so the first thing it
declined to try was the largest, which is usually the answer.

`Machine::GpuInt` takes the integers to ±2,147,483,647. A program with couplings 0.5 and 1.5 gives a
ceiling near 7.2×10⁸, so `scale_to_fit` refused a program that `s = 2` scales perfectly onto 1 and 3.
Two shipped fabric descriptors have ranges wide enough to trip it, and `Fabric::check` names
`scale_to_fit` in the message it hands callers with fractional coefficients — so the advice led
straight into a dead end. The work is bounded now, not the answer.

### Clippy had never linted five of the six crates

`cargo clippy --workspace --all-targets` **exits 101**: `src/ffi.rs` has 93 `extern "C"` functions
that trip the deny-by-default `not_unsafe_ptr_arg_deref`, and the run aborts there. Measured
consequence: **zero** lints reported for `-gpu`, `-meter`, `-cloud`, `-serve`, `-silicon` or any of
the 21 examples. 39 warnings were invisible, including `-silicon` calling its own deprecated
functions.

Suppressing a lint to make a run go green is the move this project distrusts, so the property the
lint points at was audited before allowing it: every handle goes through `as_ref`/`as_mut` (39 + 37
sites, null-checked by construction), every caller-supplied out-pointer is null-checked before it is
written, every copy is clamped with `.min(cap)`, and every `from_raw_parts` follows a null check on
the same pointer. What remains — a non-null dangling pointer, or a lying capacity — is the C ABI's
own contract, and marking these `unsafe fn` would move that obligation to callers who are C, Python
`ctypes`, Julia `ccall`, Zig and JavaScript, none of which have Rust's `unsafe` to move it to. The
allow is scoped to that module and the reasoning is written where the next reader will find it.

### Also

- `Slot::add_penalty`'s `#[must_use]` — whose note records that discarding it shipped a k=6 binary
  variable with invalid codewords costing exactly what valid ones cost, for three releases — was
  being discarded in a test helper. Asserted now.
- `-silicon`'s tests called its own deprecated `pbit_*` aliases, exercising the shim by accident and
  the real function not at all. They call the current names, and one deliberate test covers the
  alias's actual promise: that a 0.1.0 caller keeps building.
- The mutation suite is at **five** rows; the two new defects are in it.


### The Jetson meter backend, with the honest split stated

`ferrotherm-meter` has had one backend and a comment where the second should be: *"the INA3221 rails
are the equivalent. Not implemented here — the Jetson on this tailnet has been offline, and a backend
nobody can run is a backend nobody has tested."* The doctrine is right and the conclusion was too
strong. **Most of an INA3221 backend is testable without an INA3221**, because the interface is a
directory of files.

`meter::ina3221` reads them. Rail discovery, label matching, both driver layouts, both unit
conversions, the refusals and the arithmetic are covered by seven tests against fixture directories
the tests build themselves — `Rails::at` takes a path precisely so that is possible. **No reading in
it has come from real hardware**, and that is stated in the module docs rather than left to be
discovered. The part that was going to be wrong is the part that is now tested.

**The defect it exists to avoid: the rails do not add up.** On a Jetson the three channels are
*nested*, not disjoint — `VDD_IN` is the whole board and `VDD_CPU_GPU_CV` and `VDD_SOC` are parts of
what it already counts. Summing them, which is the obvious move and what a backend written from the
attribute names alone would do, roughly doubles the answer and does it silently. So it reads the
labels and uses **one** rail, and when no label says which is total it refuses and lists what it
found — there is no safe guess, since picking the largest fails exactly when a subsidiary rail
spikes, and summing is wrong always.

Two layouts, because the units differ and a factor-of-1000 slip still looks like a plausible
wattage: upstream `hwmon` exposes **no power attribute at all**, only `in[123]_input` in mV and
`curr[123]_input` in mA, so watts are `mV × mA / 1e6`; the L4T downstream driver reports mW directly.

One bug found in it while writing, by reading back what I had written: the directory walk carried a
single counter incremented per directory popped and called it depth, so it stopped descending after
six directories however shallow they were — and `/sys/class/hwmon` is shallow and *wide*, which is
exactly the shape that hid a device behind a handful of siblings. Depth is carried per entry now,
with a separate absolute bound on directories visited, and `symlink_metadata` rather than `is_dir()`
so a link pointing back up `/sys` is not followed. Covered by a test with the device behind nine
siblings.


### The mutations are written down and run every time

`mutation-check.sh` breaks one line and asks whether a named test notices. It works, and it takes
five arguments — so every mutation ever run through it was typed at a prompt once and then lost.
"These tests have teeth" was established at a moment and never re-established, while the code
underneath moved for months.

`scripts/mutation-suite.sh` is the same tool with the mutations recorded, and it is a CI job. Each
row is an invariant this project has actually got wrong, the smallest edit that reintroduces the
error, and the test that must go red: the invented LP upper bound that forced 0.12.0, the `HashMap`
whose iteration order used to decide CSR neighbour order and with it the last bits of every energy,
and the `+ 0.0` that stops `Sum for f64` folding a clean model's soft cost to `-0`.

The reason it exists is the 0.12.0 defect. 320 tests were green, and they were green because every
bound test used the two-sided `10 <= t <= 20` form, so **not one of them could distinguish the right
answer from the wrong one**. A suite is only evidence about the cases it can tell apart, and the only
way to find out which those are is to break the code and watch.

A row whose pattern no longer matches is a **failure, not a skip** — a mutation that silently stopped
applying has been reporting a pass over nothing, which is the same shape as the three other blind
checks found this week.


### You bring your own credentials, and the crate says how to get them

`ferrotherm-cloud` shipped without telling anyone how to obtain an account. The driver was correct —
it has always taken a token from the caller — but "how do I get one" was answerable only by reading
the source, and a driver for someone else's hardware that does not explain how to get access to that
hardware is not finished.

The crate docs and a new `cloud/README.md` now carry the actual process: request a token at
<https://annealing-cloud.com/en/web-api/token-request.html>, which asks for an email address and a
country and requires agreeing that you will not use the site or its output data for any purpose
including the development of weapons of mass destruction (their Terms of Use, Section 8, Export
Controls) and that you consent to the collection of personal information under those Terms. The
administrator emails the token back. That page does not state issuance time or usage limits; the
service homepage describes the Web API as free.

**And the destination is now the caller's choice.** The endpoint was written into the call site, so
the one address this crate could reach was the library's decision. It is a field with a stated
default (`ACW_ENDPOINT`), replaceable via `Hitachi::with_endpoint` and readable via
`Hitachi::endpoint`. That also makes `Device::run` reachable by a local mock, which is why nothing
had ever tested it.

A test pins the guarantee rather than leaving it to a reading of the code: with `ACW_TOKEN` unset
there is no device at all, and pointed at a port nothing can listen on, describing the fabric and
laying a program out both still succeed — because those paths are local. There is exactly one
network call in the crate, and reaching it takes a token you supplied, a program you laid out and a
call you made.

### Two crates.io pages were blank

`ferrotherm-gpu` and `ferrotherm-meter` had no `readme` key and no README, so their crates.io pages
showed nothing at all. Both have one now: the GPU's three-API verification table with the WARP
asterisk stated, and the meter's four refusals, which are the part worth reading, because the
failure mode of energy measurement is a confident number rather than an error.


### CI had been red for weeks, and the gate was hiding why

`check-semantics.sh` reported `python PRODUCED NOTHING -- its toolchain is present, so it broke` on
every CI run going back weeks, and every run of it by hand passed. Both were correct.

`cargo run --example`, which the script uses to build its reference, links the library as an **rlib**
— it does not emit the **cdylib**. On any developer machine one is already sitting in
`target/release` from an earlier `cargo build`, so the two bindings that `dlopen` the shared library
found it. On a clean runner nothing had built it, because this script never did.

The discarded stderr said, in full: *"could not load the ferrotherm shared library. Build it with
`cargo build --release`."* The fix was in the message, and `2>/dev/null` threw it away on all four
bindings. A gate that prints a verdict without the reason is a gate you cannot act on, so it now
builds the cdylib first, and prints whatever a failing binding wrote to stderr.


### check-versions now looks outside the directory

Every gate in this repository compared the repository against itself, which is why all of them were
green throughout the period `ferrotherm-gpu` sat at 0.2.0 in the tree and 0.1.0 on crates.io. The
bump was committed, changelogged and pushed. `cargo add ferrotherm-gpu` still gave you 0.1.0 and a
`Gpu` with no `is_hardware()`. Nothing was inconsistent — the tree agreed with itself perfectly. It
simply was not shipped, and nothing could see that.

`check-versions.sh` now queries the crates.io sparse index for every workspace member and fails when
a version is ahead of the registry **on a commit already pushed to main** — narrow on purpose, since
between a bump and `cargo publish` every crate is ahead and a gate that fires there is a gate people
learn to ignore.

It found two things on its first run: `ferrotherm-gpu` 0.2.0 unpublished, and `ferrotherm-cloud`
0.1.0 — the Hitachi CMOS annealing driver, carrying a full description, licence, keywords and
categories — never published and mentioned in no README or changelog.

Writing it reproduced the failure it was written to catch, twice. A never-published crate answers the
index with 404, `curl -f` calls that failure, and the first cut read `ferrotherm-cloud` as "the
network is down", stopped, and reported "publish state not checked" without ever reaching
`ferrotherm-gpu`. Then the summary line printed the all-clear above a table showing two crates that
were not on crates.io. A gate that cannot distinguish *absent* from *could not look* reports the
reassuring one; a summary that contradicts its own table teaches people to read only the summary.


### DX12 checked — for correctness, and the limit named

"Runs on Vulkan, Metal or DX12" was verified on two of the three. Now three, with an honest asterisk:

| | adapter | API | tests |
|---|---|---|---|
| Apple M5 Max | IntegratedGpu | Metal | 6/6 |
| NVIDIA L4 (EC2 `g6.xlarge`) | DiscreteGpu | Vulkan 1.4 | 6/6 |
| Microsoft Basic Render Driver (EC2 Windows) | **Cpu** | DX12 | 6/6 |

All three reproduce the exact mean energy from variable elimination. **The DX12 row is WARP, a
software rasteriser** — it establishes that the shader compiles under DX12 and that the physics is
right, and says nothing about DX12 on hardware. `Gpu::is_hardware()` reported `Cpu` and the benchmark
refused to quote a speedup, which is the guard doing its job rather than a caveat added afterwards.

Getting there took three dead ends worth recording: Windows `Write-Host` does not reach
`get-console-output`, so the reporting channel that works on Linux does not exist there (SSM does);
the base Windows AMI has no MSVC linker; and the GNU toolchain needs a MinGW `dlltool` it also does
not have. The Build Tools install is the path.


## 0.11.1 (2026-08-16)

**The OMMX export constant was documented backwards.** `Export::constant`, `ft_model_ommx_constant`
and the Python, Zig and Julia equivalents all told a caller to ADD the offset to the OMMX objective
to recover ferrotherm's energy. The exporter already applies it — it is written into the instance's
`Linear` message — so `ommx_objective(x) == ferrotherm_energy(s)` exactly, and a caller who followed
the documentation got a number wrong by precisely that constant.

The code was right the whole time. So was the reference test, which compares the two objectives
**without** adding anything and therefore agreed with the code rather than the prose. Nothing failed;
five surfaces carried the same wrong sentence for one release.

Now pinned by `an_exported_objective_needs_no_correction`, which asserts that adding the export
constant makes the comparison *wrong* — so if the exporter ever stops folding it in, the test says
the documentation is stale rather than the arithmetic.

## 0.11.0 (2026-08-16)

**Reproducible, measured, and speaking the format the field converged on.** Breaking:
`requires-python` is now `>=3.11`, `Ledger::joules` returns `Option<f64>`, `Prices` gained a
`source` field, and eleven functions returning a decision are `#[must_use]`.

The headline is a correctness fix. **"Deterministic by seed" was only half true** — a `HashMap` in
`GraphBuilder::build` randomised the CSR neighbour order, so the same model compiled to five
different programs across five runs and a derived energy took six distinct values from one identical
state. One word (`BTreeMap`) fixed it, and it is now measured on three machines rather than asserted
on one.

Also new: `ferrotherm-meter` (measured wall power), `src/ommx.rs` (the interchange format, both
directions, all nine surfaces), constraint detection, and three checks — `check-semantics`,
`check-exports` and the sibling-crate rule — each written after the bug it would have caught.

### OMMX import reaches the bindings too

Export landed on all nine surfaces; import stayed Rust-only, which is the same asymmetry in
miniature. `ft_ommx_read` / `ft_ommx_error` (C ABI and header), `ft.from_ommx()` (Python),
`ommxRead` / `ommxError` (Zig), `from_ommx` (Julia).

Verified end to end rather than by shape: the Python binding reads an instance the reference stack
built, anneals it, and lands on **the reference's own optimum** (state `1110`) with the energies
agreeing exactly. Julia does the same. A continuous variable is refused by name — *"decision
variable 0 ('temperature') has kind 3; ferrotherm samples spins"*.

### `check-exports.sh` — a name exported with nothing behind it

Writing the above, both Julia OMMX functions ended up **declared and exported with no function body
in between**. The module loaded cleanly, `check-parity.sh` passed — it matches the `@cfn`
declaration, which says a symbol is callable and nothing about whether anything calls it — and
`from_ommx` was an `UndefVarError` the first time it was used. Julia does not resolve an exported
name until then.

The new check asks the interpreter to resolve every exported name. It is the cheapest check in the
repository and it would have caught this immediately; mutation-tested by exporting a name that does
not exist. `check-parity.sh` now says in its own comments what it does and does not prove.


### `check-parity.sh` now sees the gap it could not

Everything that check did asked whether the core's C ABI reaches the bindings. It had nothing to say
about a capability that never entered the core at all — which is how the OMMX bridge shipped as a
sibling crate, reachable from Rust and none of the other eight surfaces, **past the check written
precisely to stop that**.

A sibling exists because the core cannot hold it, and every real one can say why:

| | why it is out |
|---|---|
| `silicon`, `cloud`, `gpu` | an external dependency |
| `serve` | an application — two binary targets |
| `meter` | `std::process`, which the core's wasm target does not have |

A sibling with none of the three has no reason to be one, and being out there costs it eight
surfaces. `ommx` had zero external dependencies, zero binaries and no wasm-hostile API. The check now
fails on that shape; mutation-tested by recreating the crate exactly as it first shipped.

Three bugs in writing it, all the same one: under `set -euo pipefail` a `grep` that matches nothing
returns 1, pipefail propagates it, and `-e` kills the script *after* the assignment has succeeded.
The symptom is a heading printed with nothing beneath it — a check that looks like it ran.


### OMMX on all nine surfaces — and moved into the core, where format bridges live

The bridge shipped as a sibling crate, which was inconsistent with this codebase's own structure:
`src/ftp.rs` and `src/lp.rs` are both format bridges and both live in the core. A sibling meant the
C ABI could not reach it and **eight of the nine surfaces could not export at all**. Now `src/ommx.rs`,
beside the two it belongs with, and reachable everywhere:

`ft_model_ommx` / `ft_model_ommx_constant` (C ABI and header) · `Problem.ommx()` (Python) ·
`ommx` / `ommxConstant` (Zig) · `ommx(p)` (Julia) · `ommx_b64` + `ommx_constant` (HTTP, MCP) ·
byte count in the editor.

A Python user can now hand a ferrotherm model straight to the OMMX ecosystem — verified end to end:
the binding exports 373 bytes and the reference `Instance.from_bytes` reads 6 variables at degree 2.
The HTTP payload is checked to decode to **the library's own bytes**, so the wire path cannot
diverge from the in-process one.

Base64 in the JSON surfaces is hand-rolled: `serve` has no dependencies and one encoder does not
justify the first.

### `src/ommx.rs` — a `.ftp` program the rest of the ecosystem can read

OMMX is the interchange format this corner of the field has converged on: jijmodeling 2.x compiles
to it, and it is a shared dependency across the Jij stack. A compiled ferrotherm program now exports
as an `ommx.v1.Instance` — binary decision variables, a quadratic objective, `SENSE_MINIMIZE` —
which is a lossless target for an Ising model.

**Zero dependencies.** The Rust `ommx` crate is `3.0.0-beta.3` while its Python counterpart is stable
at `2.6.2`, and a shipped bridge should not rest on a beta. The subset needed is varints and
length-delimited fields, and the field numbers were read out of the reference implementation's own
protobuf descriptors rather than from prose.

**Validated by the reference implementation, not by my reading of the schema.** The unit tests here
check my own arithmetic and would pass just as happily with every field number wrong, because they
never leave the process. So `tests/reference.rs` shells out to Python's `ommx` — different language,
different codebase, the format's own maintainers — and has it evaluate the instance:

> **512 states scored by the reference implementation, 0 disagreements with ferrotherm.**

The substitution is the one thing to know: ferrotherm's spins are ±1 and OMMX binaries are 0/1, so
`s = 2x − 1` is applied during export. That changes every coefficient and introduces a constant; an
exporter that skipped it would produce a file that parses cleanly and describes a different model.
The constant is carried in the `Linear` message so the OMMX objective equals ferrotherm's energy
exactly rather than up to an offset a reader has to know about.

**Import works too**, which is what makes this a bridge rather than an exporter: ferrotherm reads an
instance the reference stack built end to end, and scores all 16 states of it identically. What it
cannot sample is refused **by name** — a continuous variable ("no spin encoding at any width"), a
non-[0,1] bound, an objective of degree ≥ 3 (pointed at `ferrotherm::reduce`, so the ancilla count
is visible rather than paid silently). Repeated scalar fields are accepted packed *or* unpacked;
the reference packs, and this crate's own encoder did not until it was checked against real output.

**Reading a real file found a bug my own round-trip could not.** proto3 omits a field at its default
value, so `Term { id: 0, coefficient: c }` serialises the coefficient alone — and the reader used a
sentinel for "no id seen", turning variable 0 into "not declared". This crate's encoder writes id 0
explicitly, so its reader and writer agreed with each other and were both wrong about the format.
Only somebody else's encoder could show that.


### Constraint detection — reported, never rewritten

jijmodeling 2.x recognises one-hot patterns and hints the solver. Same idea, different verdict:
`Compiled::caveats` now names a constraint written in a form that measures more expensive, and
**leaves the model alone**. Silently compiling something other than what was written is the opposite
of this compiler's discipline, and a modeller who meant the expensive form is entitled to it.

**The gap turned out to be much smaller than inferring suggested**, which is why it was measured:

| longhand | direct | verdict |
|---|---|---|
| `cardinality(lits, 1)` — 10 spins, 15 factors | `exactly_one` — 10, 15 | identical, no caveat |
| 6× `not_equal` — 16 spins, 48 factors | `all_different` — 16, 48 | identical, no caveat |
| `at_most(lits, 1)` — **12 spins, 26 factors** | `at_most_one` — **10, 15** | a real saving |

Only the last earns a caveat: an inequality has to become an equality the sampler can square, and at
k = 1 the pairwise exclusion says the same thing for free. Advice that costs a reader time and saves
no spins is noise, and a checker people learn to ignore catches nothing.

Also detected: constraints that constrain nothing — `at_most(n of n)` and `at_least(0 of n)` are
satisfied by every assignment and still pay for a slack variable and its factors. Almost always a
`k` that was meant to be different.

`docs/LANDSCAPE.md` is corrected accordingly: yesterday's entry said a longhand modeller "gets the
expensive lowering and no warning", which was inference. Measured, it is true of one form out of
three.


### Python floor raised to 3.11, and why it matters beyond packaging

`requires-python` said `>=3.9` while CI tested exactly one version. The floor was a claim nobody had
ever run — and 3.9.6 is what macOS ships and nobody chooses. Now `>=3.11`, tested on 3.11, 3.12 and
3.13 in a CI matrix rather than at one point.

This was not a packaging detail. **The landscape survey was run on 3.9.6**, and three of the packages
it assessed cannot install their current versions there: `jijmodeling 2.7.1` needs ≥3.11, `amplify`
and `ommx` need ≥3.10. pip resolved the newest 3.9-compatible release and the survey recorded those
as the state of the art. An obsolete interpreter turned into a wrong competitive assessment.

### Landscape re-survey on 3.13 — two real gaps found

8 of 8 spot-checked packages had moved, one by a major version. Two findings change our position:

- **jijmodeling 2.x detects constraint patterns and we do not.** `ConstraintDetectionConfig` /
  `ConstraintHintName` (`OneHot`, `Sos1`) recognise that a set of constraints forms a one-hot
  pattern and hint the solver. Ferrotherm requires the modeller to write `exactly_one` and rewards
  them with a cheaper lowering — but writing the pattern longhand gets the expensive lowering and no
  warning. A real gap in the modelling row.
- **Amplify ships 16 vendor clients to our 7 fabrics.** Broader vendor reach than ours. Checked by
  introspection rather than assumed: it has no energy, joule, watt or power surface, and no
  certificate surface. Those rows still stand.

Also: jijmodeling 2.x compiles to **OMMX** and round-trips protobuf. Jij has committed to OMMX as the
shared IR, which is the context `.ftp` sits in and worth a deliberate decision.


### Semantic parity: seven surfaces, one model, one hash

`check-parity.sh` proves every C ABI symbol reaches every binding. It says nothing about whether
they **compute the same thing** — nine surfaces can each have `all_different` and three can get it
wrong. `scripts/check-semantics.sh` builds one model through Rust, Python, Zig, Julia, HTTP, MCP and
the wasm editor, and byte-diffs the compiled `.ftp` program, which is the semantic fingerprint.

**All seven emit identical bytes**, `7a769648af237e11`, 3103 each. HTTP and MCP are driven through
their real binaries; the editor through a headless browser.

The model is chosen to exercise what has been wrong before: an integer over `10..=13` so
`fix(t, 12)` must mean *twelve* and not slot twelve, a counting constraint that costs a slack
variable, and objective terms carrying different weights *and* different values.

Both mutations it was tested against are caught — a wrong objective weight in the HTTP path (HTTP
and MCP diverge together, as they share a dispatch), and the historical slot-for-value bug in
Python.

**The first mutation initially slipped through.** Python raised, produced no file, dropped out of
the comparison, and the run reported "5 bindings identical" — true, and useless. A binding whose
toolchain is *absent* now skips; one that is *present and produces nothing* fails, because that
means it broke.


### The GPU backend verified on a second vendor and API

"Runs on Vulkan, Metal or DX12" was a claim checked on **one** of the three. Now two:

| | adapter | API | tests |
|---|---|---|---|
| Apple M5 Max | IntegratedGpu | Metal | 6/6 |
| NVIDIA L4 (EC2 `g6.xlarge`) | DiscreteGpu | Vulkan 1.4.329 | 6/6 |

Both run the same WGSL from the core crate and both reproduce the **exact mean energy computed by
variable elimination** — not agreement with each other, agreement with the physics. A shader can
pass on Metal and fail on Vulkan, whose validation is stricter, so this was worth checking rather
than assuming. DX12 remains unchecked and is written down as such.

On the L4 the throughput curve is steeper than on the M5 Max — 80× at 262k nodes against
single-threaded CPU, versus 31× — which is what a discrete card with its own memory should do. The
same caveat applies: that column is one CPU core, and the fair multi-core figure is roughly a
quarter of it.


### Determinism, measured on three machines

Yesterday's `BTreeMap` fix made the stack byte-reproducible **on one machine**. That is a weaker
claim than it sounds, so it was checked on two more: Linux/x86_64 (AMD EPYC 9R14) and
Linux/aarch64 (Graviton3), against macOS/arm64 (Apple M5 Max), all on rustc 1.97.1.

| | macOS arm64 | Linux x86_64 | Linux aarch64 |
|---|---|---|---|
| compiled `.ftp` program | identical | identical | identical |
| CSR neighbour order | identical | identical | identical |
| **sampled state** | identical | identical | identical |
| `exp()` and the sigmoid | identical | identical | identical |
| energy from that state | `…a7b3` | `…a7b2` | `…a7b2` |

**The answer is bit-reproducible across OS and architecture.** A derived float can differ by **one
ULP across operating systems** — and the cause is not what it looks like. The two Linux boxes agree
with each other on *different* architectures while macOS disagrees with Linux on the *same* one, so
it is not architecture; and `exp` was measured bit-identical on both, so it is not libm. It is
floating-point contraction: `w * s_i * s_j` going through an `fma` on one target and a separate
multiply and add on another.

Documented at the top of the crate: compare states, hashes and programs with `==`; compare energies
with a tolerance.


### "Deterministic by seed" was only half true

`GraphBuilder::build` merged duplicate edges through a `HashMap`. Rust randomises HashMap iteration
per instance, and that order decides the CSR neighbour order — which decides the order every local
field is **summed** in. Float addition is not associative.

Measured over eight builds of one graph, before the fix:

- **8** distinct CSR neighbour orders
- **1** sampled state — the RNG stream does not depend on the order, so the answer was stable
- **6** distinct energies computed from that identical state, every one printing the same, because
  they differed in the last bits

And `Program::to_ftp` was not reproducible: five runs of one model emitted five different programs —
pure permutations of each other, identical in length, which is why nothing noticed. A program IR
whose bytes depend on which run produced it cannot be hashed, diffed, cached, or checked for
reproducibility.

One word: `BTreeMap`. The merge goes from O(m) to O(m log m), which is nothing beside the sampling
it feeds, and the whole stack becomes byte-reproducible.

**How it was found.** `check-parity.sh` proves a symbol exists on nine surfaces. It says nothing
about whether they compute the same thing — so I built one model through Rust and through Python and
diffed the compiled programs. They disagreed. Symbol parity is not semantic parity, and the same
model now compiles byte-identically through both.


### The GPU is 12x faster and 10x cheaper — not 54x

`gpu/examples/joules.rs` measures **both** paths at the wall on one machine and divides by the node
updates the ledger counted. Apple M5 Max, 512×512, 4-second windows:

| | throughput | J per node update |
|---|---|---|
| GPU | 5.98e9 updates/s | **3.90e-9** |
| CPU, all 18 cores | 4.98e8 updates/s | **3.92e-8** |

**12.0× faster, 10.0× cheaper per update.** The speedup exceeds the saving, which is the expected
shape: the GPU draws more power while it works, and time and joules are different questions.

`bench.rs` reported **54×**, and that number compares a whole GPU against **one CPU core of
eighteen** — the oldest way to flatter a GPU benchmark. `Sampler::sweeps` is single-threaded;
`sweeps_par` is the fair comparison and the ratio drops by a factor of four. The bench now says so
at the bottom of its own output rather than in an errata.

Three things this measurement needed, each found by it failing:

- **Repeat until the instrument can see it.** 157M GPU updates finish in ~30 ms, so a single run
  left the measurement window almost entirely idle and reported 1.26 W above a 1.10 W baseline
  wander. You cannot measure something faster than your sampling interval by running it once.
- **Cool down between paths.** The CPU's baseline taken straight after a 4-second GPU burn wandered
  by 7.2 W, which swallowed the CPU's own signal. Two workloads measured back to back are not
  independent unless the machine is allowed to forget the first.
- **Use the machine fairly, which is also the only way to see it.** One busy core added 0.13 W to a
  baseline wandering by 1.43 W. Using all eighteen made the comparison honest and the signal
  measurable in the same change.


## 0.10.0 (2026-08-15)

**Every capability the landscape survey measured, and one side of the energy claim now measured
rather than borrowed.** Breaking: `Ledger::joules` returns `Option<f64>`, `Prices` gained a `source`
field, `Constraint` and `CompileError` gained variants. Two new sibling crates: `ferrotherm-gpu`
(native GPU sampling) and `ferrotherm-meter` (measured wall power).

This release closes the last four rows in `docs/LANDSCAPE.md` where any surveyed competitor —
Extropic, D-Wave Ocean, Fixstars Amplify, Jij, PyQUBO, the open-source Ising layer, or the hardware
vendors' own SDKs — had something this stack did not. Nine of the fifteen capabilities are ones no
other surveyed stack has at all.

### `ferrotherm-meter` — measured wall power, so a joules figure describes the machine that produced it

The ledger counted operations exactly and priced them against the one table in the tree: `Z1_SPICE`,
pre-silicon estimates for an accelerator nobody has characterised. `Prices::UNSTATED` made borrowing
it honest. This makes borrowing it unnecessary.

- **Measured on an Apple M5 Max: 4.261e-7 J per node update** — whole-system wall power above idle,
  over an 8.29 s window with 75 power readings. `Z1_SPICE` estimates 7.09e-15 J. The ratio is 6.0e7,
  and it is *the size of the prize being claimed, not a measured speedup*: one side is a
  general-purpose CPU at the wall, the other a per-device SPICE estimate for silicon that has not
  been fabricated. One side is now measured.
- Std-only, no dependencies — the backend is a subprocess (`macmon`), and one field out of a JSON
  line does not justify a parser.
- **Four things it refuses**, each found by it happening: a workload too short for the backend's tick
  (a mean of two readings is not a mean); a run drawing *less* than its baseline (which means the
  baseline was taken on a busy machine — the first version clamped this to zero and thereby reported
  that computing is free); a delta inside the baseline's own 3σ wander (two runs of one workload
  reported 2.34e-8 and 1.81e-7 J/update, an 8× spread, because one added ~1 W to a ~60 W baseline);
  and a mixed workload turned into a per-sample price (three unknowns, one equation).
- Settling is part of the protocol, measured rather than guessed: a baseline taken right after a
  heavy run read 68.5 W while the run itself averaged 64.1 W. Fans and thermal management lag the
  workload, so idle straight after load is systematically *higher* than idle.
- Jetson/Linux INA3221 rails are the same measurement and are **not implemented**: that host has been
  offline for a week, and a backend nobody can run is a backend nobody has tested.


### A joules figure now has to say whose machine it describes

**Breaking:** `Ledger::joules` returns `Option<f64>`, and `Prices` gained a `source` field.

Every fabric in the tree declared `Z1_SPICE`. So a Hitachi CMOS annealing ASIC and a laptop CPU both
reported Extropic's pre-silicon SPICE estimates as their own energy, and the HTTP surface reported
them for every run including a plain CPU sample. Nothing was lying — nothing had been asked to say
whose numbers these were.

- `Prices::source` travels with the numbers. `Prices::UNSTATED` is what "nobody has published this"
  looks like in the type system; Hitachi and the CPU device declare it.
- `joules` returns `None` rather than zero for an uncharacterised device. Zero is a claim that it
  costs nothing.
- HTTP reports exact **counts** always, `joules` as null when unpriced, and `priced_as` generated
  from the prices — a hardcoded note is how a figure ends up labelled with the wrong machine.

### The write is charged

`Ledger::writes` was incremented by **nothing** in the library, so every figure the stack produced
was a sample-and-read story with the expensive term silently zero — on this hardware class a write
costs roughly 21,700 samples, which is the module's entire thesis. `Device::program` *is* the write;
the trait says so now, and the CPU and Hitachi implementations charge one per node. A demonstration
run reports the write at **100%** of the projected energy.

### `reflash_hz_cap` is read

It was declared and consulted by nothing. `Ledger::reflash_seconds` turns it into a wall-clock floor:
a workload that reflashes the whole graph faster than the device sustains is not fast, it is
unphysical, and pricing it describes a run that could not have happened.


### The compiler says what it knows

`Slot::add_penalty` has always returned whether the encoding it added can be exact, and **both
callers threw it away**. `Compiled::caveats` now carries it, on every surface.

- A binary encoding of *k* values uses ⌈log₂ k⌉ spins, spelling 2^⌈log₂ k⌉ codewords. When *k* is not
  a power of two the spare codewords decode to nothing — and **no penalty removes them**. Measured
  on k = 6: the cheapest invalid state costs `0.00` and so does the cheapest valid one, so the
  sampler has no reason at all to prefer an answer. A test enumerates all eight codewords, so the
  message cannot drift from the physics it describes.
- `ft_model_caveats`/`ft_model_caveat` (C ABI and header), `Answer.caveats` (Python),
  `Problem.caveats`/`caveat` (Zig), `caveats(a)` (Julia), `"caveats"` (HTTP, MCP), and the editor,
  where a caveat now outranks "every constraint holds" in the status line — that is exactly the case
  where the answer looks clean and the model is not.

### Encoding selection reaches the last two surfaces

HTTP and the node editor **silently ignored `"encoding"`**: a document asking for binary got one-hot,
with a different spin count, a different penalty and no error — the reply's own `ftp` said `onehot`
and nothing else did. Both now honour it and refuse an unknown name by listing the ones that exist.


### Native GPU sampling — `ferrotherm-gpu`

The last row in `docs/LANDSCAPE.md` where a surveyed competitor had a sampler this stack did not.
GPU sampling existed only in the browser; the native core was CPU-only.

- A **separate crate**, beside `silicon`, `serve` and `cloud`. `ferrotherm` stays std-only with zero
  dependencies, which is what lets the same source compile to `wasm32-unknown-unknown` and to a
  microcontroller — a property worth keeping rather than a slogan.
- It has **no shader of its own**. The WGSL comes from `ferrotherm::wgsl::sweep_shader`, the same
  string the browser fetches through `ft_shader`, so a native run and a browser run cannot disagree
  about the arithmetic — only about the hardware.
- **Measured on an Apple M5 Max (Metal), median of 5 runs, 200 sweeps:** crossover at ~4k nodes,
  3.5× at 4,096, 14× at 16,384, 42× at 65,536, 54× at 262,144. The GPU **loses** below ~1k nodes and
  the benchmark says so: fixed cost per run does not shrink, so the crossover is the number worth
  quoting rather than the peak.
- `Gpu::is_hardware()` and the benchmark refuse to quote a speedup against a software rasteriser.
  lavapipe, SwiftShader and WARP all run this shader correctly and say nothing about a GPU.
- A readback that is not ±1 is **refused, not coerced** — the same discipline the browser path
  learned, where `> 0 ? 1 : -1` laundered a dropped dispatch into a believable energy.

### Fixed before it shipped

- **One command submit per dispatch made the GPU slower than the CPU at every size**, at a near
  constant ~60 ms regardless of node count — constant time under a growing workload is the signature
  of paying for driver round trips rather than arithmetic. Now one encoder, one pass and one submit
  for the whole run, with per-dispatch parameters selected by dynamic uniform offset. 60 ms → 2.5 ms,
  and the ratio at 65k nodes went from 0.75× to 42×.
- `Limits::downlevel_defaults()` caps storage buffers at 4 per stage and this shader binds 6, so the
  device compiled nothing. The WebGPU baseline is the right floor: anything that runs the page runs
  this.


### all-different, the constraint no pair of variables can express

`Model::all_different` (Rust), `ft_model_var` + close kind 5 (C ABI and header), `all_different`
(Python), `allDifferent` (Zig), `all_different!` (Julia), `{"type": "all_different", "of": [{"var":
name}, ...]}` (HTTP, MCP, node editor). The last gap in the constraint vocabulary against every
other modelling layer surveyed in `docs/LANDSCAPE.md`.

- **Lowered per shared value, not per pair.** For each value two of them could both take, the
  indicators are excluded pairwise — the `AtMostOne` lowering repeated over shared values. No slack,
  no ancillas, and *nothing at all* where the domains do not overlap: `all_different` over `0..=3`
  and `10..=13` adds zero couplings, which a sweep of n(n−1)/2 `not_equal`s would not notice.
- **The violation names which value collided and who took it** — "a and b both take 2" is a repair;
  "all-different was violated" is something the modeller already suspects.
- **The pigeonhole case is refused at compile time, by name.** More variables than the values they
  share between them has no answer at any penalty. Annealing it and reporting `feasible: false`
  reads as "raise the penalty" or "lengthen the ladder", and neither can work. Counting is cheap, so
  it is counted.
- `ft_model_var` appends a variable rather than a literal, picking a value from the variable's own
  domain. The first version passed a placeholder `0` through `ft_model_lit` and was correctly
  refused for every variable whose domain did not contain 0 — that function's whole job is to reject
  a value a variable cannot take, so the fix belonged in the library, where the domain is known.


## 0.9.0 (2026-08-15)

**A constraint can be a price, and surface parity is checked rather than remembered.** Breaking:
`Violation` gained fields, Julia's `Answer` gained three, and Zig's `Error` gained `BadState`.

### Soft constraints — a preference, priced, on every surface

A constraint has always been either a rule or nothing. A soft one is a **preference the solver may
trade away**: breaking it costs, and the answer stays feasible. The distinction is not a shade of
penalty strength — it changes what the answer *means*. A broken rule says the answer is not an
answer; a traded preference says the solver made the choice it was asked to price.

- `Model::soft(c, weight)` and `Model::soften_last(weight)` (Rust), `ft_model_close_soft` /
  `ft_model_soften_last` / `ft_model_soft_cost` / `ft_model_violation_is_hard` (C ABI and header),
  `soft=` on every constraint plus `soften_last` (Python), `countSoft` / `softenLast` / `softCost` /
  `violationIsHard` (Zig), `soft =` keyword plus `soften_last!` / `soft_cost` / `traded` (Julia),
  `"soft": w` on any constraint (HTTP, MCP and the node editor). Eight surfaces, one release.
- `Solution::feasible` ignores soft violations by design; `Solution::soft_cost` totals what the
  trades cost, and each `Violation` carries `hard` and `cost` so the two are never confused.
- **The price is `weight × amount²`.** A constraint becomes an energy term by squaring how far
  outside it sits, so missing by two costs *four* times missing by one. Pricing a preference chooses
  that curve as well as its scale, and reporting a linear price would misstate what was traded.
- The weight is **absolute, not scaled**. Automatic penalty scaling exists to stop a hard constraint
  being outbid by the objective; a soft one is meant to be traded against it.
- A `"soft"` that is not a positive finite number is refused **by name** on every JSON surface
  rather than falling back to a hard constraint — the same shape as five earlier silent-wrongness
  bugs, where an unreadable field became the opposite instruction with `feasible: true` and nothing
  to suggest anything had gone wrong. The editor checks `typeof`, not coercion, so the same document
  gets the same verdict there as over HTTP.
- The editor's status line no longer claims "every constraint holds" over an answer that traded one.

### Parity is now checked, not remembered

`scripts/check-parity.sh` asks, for each of the 84 exported C ABI symbols, whether the header
declares it and Python, Zig and Julia call it. A gap passes only with a written reason in its EXEMPT
table — which is what AGENTS.md has always meant by "the gap is written down", except that until now
nothing read it. It is in CI, and its first run found sixteen real ones:

- **`ft_model_ancillas` reached no binding.** It reports the spins the higher-order lowering added,
  which is the number that says whether *sampling* from a solved model is sound at all — the
  reduction is exact for optimisation and not for the Boltzmann distribution. Zig even had a test
  named for ancillas that could not read the count. Now on Python (`Answer.ancillas`), Zig
  (`Problem.ancillas`) and Julia (`ancillas(a)`).
- **`ft_set_spins` was written, unit-tested and called by nothing.** It puts a state computed
  elsewhere into a simulation so the same code scores it — and it refuses a wrong length or a value
  that is not ±1. The browser's WebGPU path wrote straight into wasm memory after coercing every
  value with `> 0 ? 1 : -1`, which launders a dropped dispatch or a short readback into a plausible
  state that is then scored with confidence. The page now goes through the validation; `sim.spins =`
  (Python), `setSpins` (Zig) and `spins!` (Julia) expose it.
- **Violation magnitude never reached Zig or Julia.** `violationAmount` and `amounts(a)`: "at most
  two, and four hold" is over by two, which is a different problem from being over by one.
- **A solved `Problem` could not be certified from Zig.** `Problem.certify` and
  `ProblemCertificate`, separate from the simulation certificate because the exact bounds are not
  available there and half a struct of NaNs would claim they were computed.
- **`exact_ground_state` was Julia-only.** Python and Zig had the ground *energy* and not the
  assignment that reaches it, which is what checking a sampler against the truth needs.
- The header now declares `ft_field`, `ft_shader`/`ft_shader_len` and the seven `ft_gpu_*`
  device-buffer accessors. A header is the ABI contract; selectivity belongs in the bindings above
  it.

### Fixed

- `scripts/check-versions.sh` named `serve` as the only crate pinning the library, so this release's
  bump left `cloud` and `silicon` at `0.8` while the script reported "all agree". It now *finds*
  every dependent instead of listing them, with a floor so a search that stops matching fails rather
  than passing over nothing. An exemption in `check-parity.sh` naming a symbol that does not exist
  now fails too — a table of reasons nobody can check is the thing that script replaced.
- **`ferrotherm_jll` has never been installable, and its check passed anyway.** The Julia artifacts
  were built, hashed, and uploaded to a CI artifact that expires — while the `Artifacts.toml` shipped
  in the package named `releases/download/vX/…` URLs that no release ever carried. Every one has
  been a 404 since 0.7.0. The verification step rewrote those URLs to a `localhost` server before
  testing them, so it reported on the stand-in rather than on what a user would fetch. The release
  job now creates the release and attaches the tarballs, then fetches each **published** URL and
  checks the bytes against the hash the manifest commits them to — with a floor, so finding no URLs
  fails instead of passing over nothing. The manifest is also committed back into the package: the
  one in the repository sat at 0.8.0 hashes while the library moved on.
- `Solution::soft_cost` returned `-0.0` when nothing was traded. Rust's `Sum for f64` folds from
  `-0.0`, which is the correct additive identity and prints as a minus sign through every binding
  that formats a float. A price with a minus sign in front of it reads as a credit.

## 0.8.0 (2026-08-13)

**A fabric describes a real machine, and a model can be wider than pairwise.** Breaking: the
precision field changed type, `Fabric` gained fields, `Unsupported` gained variants.

### Higher-order models

- `reduce::to_pairwise` lowers a k-body model onto pairwise hardware. An ancilla spin is defined as
  the product of two existing ones and substituted wherever that pair appears; the pair chosen each
  round is the commonest, so three 3-body terms sharing a pair cost **one** ancilla, not three.
  It goes through binary because in spin space *"t equals s_a·s_b"* is itself a three-body
  statement — Rosenberg's `3y + x_a·x_b − 2x_a·y − 2x_b·y` is quadratic throughout.
- **Verified by enumerating every state of both models.** For each assignment of the original
  spins, the reduced energy minimised over the ancillas equals the original plus one constant, so
  nothing is reordered and the ground states correspond exactly.
- **The guarantee is about optimisation.** The ancillas add states, so the Boltzmann distribution
  over the original variables is not preserved at finite temperature. Read `reduce`'s docs before
  sampling from a reduced model.
- `Model::compile` accepts a cubic or wider objective and applies the pass; `Compiled::ancillas`
  reports the price. Whatever expands to degree 0, 1 or 2 goes straight into the graph, so only
  what is genuinely wider is charged.
- Reachable everywhere: `Expr::product` (Rust), `ft_model_objective_product` (C),
  `a.is_(1) * b.is_(1) * c.is_(1)` (Python), `preferAll` (Zig), `(w, (l1, l2, l3))` (Julia),
  `"of": [...]` (HTTP/MCP), and a variadic **Prefer all together** node in the editor.

### Fabrics

Six machines declared from vendor documentation, each number cited where it is used:

| fabric | from | notable |
|---|---|---|
| D-Wave Advantage2 | topology docs, GA announcement | Zephyr, 20-way, `j_range [-1,1]` |
| D-Wave Advantage | topology docs | Pegasus, 5,640 qubits, 15-way |
| Fujitsu DA3 | Fujitsu's API documentation | 100,000 bits, **fully connected**, 64-bit integers |
| Toshiba SQBM+ (QUBO) | Toshiba's user manual | 10M variables, **float32** |
| Toshiba SQBM+ (PUBO) | same | **order 4 natively** — no reduction needed |
| QBoson CPQC | Kaiwu SDK docs | **8-bit integers, [-128, 127]** — the hardest limit here |

- `Range { lo, hi, integral }` says what magnitudes a coefficient may take and whether they must be
  whole. D-Wave's couplings are continuous over `[-1, 1]`; Hitachi's are four-bit integers over
  `-7..=7`. `J = 0.5` fits the first exactly and the second not at all, and a bit count cannot say
  that. `Fabric::scale_to_fit` returns the factor that makes a program fit, or `None` when no
  factor helps.
- `Precision::{Exact, Fixed, Float, Unstated}` replaces `Option<u32>`. Fixed point spreads one step
  across the range and loses a small coefficient; floating point keeps significant digits and does
  not. Over `[1e8, 1.0]` the two answers are `1.0` and `6e-8`.
- `Verdict` carries **every** caveat rather than the first one checked. D-Wave has two: it places by
  minor embedding *and* its precision is unpublished.
- `Fabric::unstated` names what a vendor does not publish. `None` used to mean both "no limit" and
  "not documented", so a machine of unknown size looked exactly like a simulator with no size.

### Fixed

- `feasible` now means the constraints hold, not merely that every variable decoded. A penalty makes
  a constraint expensive, not impossible; a sampler whose objective outbids it returned an answer
  that read perfectly and broke the request, reported as feasible. `violated` names what broke.
- Two variables may not share a name. An answer is keyed by name, so the second replaced the first
  and one of them vanished from the result.
- A unary factor is a **field**, as it is everywhere else in the crate. It was range-checked as a
  coupling, so fields written that way walked past a fabric that has none.
- Degree counts edges, not factor mentions. `uniform_couplings` compares values, not bit patterns,
  so `0.0` and `-0.0` are one weight rather than two.
- `check` reports the worst out-of-range coefficient, which is the one that sets the scale factor.
  It reported the first while its comment said worst.
- The node-update ceiling counts the whole run. A request declaring 1,024 node updates did 262
  million, because every handler runs `draws × thin` further sweeps afterwards.
- Every Hitachi layout failure was reported as `TooHighDegree { degree: 0, limit: 8 }`. It says what
  failed now.
- `field_bits` was declared by every fabric and read by nothing.
- The Pt V2 declared a negative coupling its cell cannot express — it counts active neighbours, so a
  coupling is present or absent.

### Verification

- `scripts/mutation-check.sh` breaks the code on purpose and requires a named test to notice. It
  refuses to run on uncommitted work, calls out a mutation that did not apply, and distinguishes a
  build failure from a genuine red — each of which had already produced a false pass here.
- `include/check.c` compiles against the header and links against the library, so the header cannot
  drift from the ABI. It found a missing declaration on its first run.
- `web-tests/` drives both browser pages through the same API an agent would use, and
  `npm run live` drives the deployed copy rather than a local build.

**403 Rust, 18 Python, 19 Zig, 123 Julia, 42 browser, 18 C.**



## 0.7.0 (2026-08-08)

**The modelling layer, on every surface.** And an ABI break to make it honest: a `value` is now
`int64_t` everywhere and means the modeller's own value, never a slot index.

### The layer

`model` states a problem in its own vocabulary — variables with domains, constraints that must
hold, an objective — and compiles to spins, answering in the names you gave.

- Variables: `categorical`, `integer`, `binary`, `spin`. Constraints: `not_equal`, `equal`, `fix`,
  `cardinality`, `at_most`, `at_least`, `exactly_one`, `at_most_one`. Objectives read like
  arithmetic: `5.0 * x.is(2) + 2.0 * y.is(1) - a.is(1) * b.is(1)`.
- Inequalities compile through a **slack variable**, because squaring "at most three" would punish
  choosing two exactly as hard as choosing four. The slack costs spins and never appears in the
  answer: a solver artefact is not a result.
- Counting constraints take **any number of literals**, each naming its own variable and its own
  value. "At most two of these nine shifts" and "at most one of a = 3, b = 17" are both sayable.
- Reachable from Rust, C, Python, Zig, Julia, wasm, the node editor, HTTP and MCP. Every one
  compiles through the same code.

### Breaking

- **Every `value` widened from `uint32_t` to `int64_t`** and carries the modeller's own value. An
  integer over `10..=20` takes 13; passing 3 is an error naming the range. It used to be a slot
  index, so `x.is(13)` rewarded **18** — and `not_equal` compared two variables slot by slot, so an
  integer over `5..=10` and one over `0..=5` were held to agree in six places when they share one.
- **`feasible` now means the constraints hold**, not merely that every variable decoded into a
  valid codeword. A penalty makes a constraint expensive, not impossible; a sampler whose objective
  outbids it returns an answer that reads perfectly and breaks the request, and that reported as
  feasible. `violated` describes each broken constraint in the caller's own names.
- `Model::objective` **accumulates** rather than replaces, and folds its sense in per term. Writing
  one term per option in a loop kept only the last; a minimising term after maximising ones
  re-interpreted all of them. `set_objective` replaces.
- `Domain::Spin` speaks in −1 and +1. It reported 0 and 1 to the literal reader while the decoder
  handed back −1 and +1, because both folded Spin into a `_ =>` catch-all.
- Two variables may not share a name. An answer is keyed by name, so the second did not shadow the
  first — it replaced it, and one of the two vanished from the result.

### Fixed, all of them silent

Each returned a plausible answer with `feasible: true` and no error.

- `"maximize": 1` **minimized**. `as_bool` returned `None` for a JSON number and `unwrap_or(false)`
  made that `Minimize` — not a degraded answer, the opposite one.
- A `"value"` the reader could not parse became 0, or 1. `"13"` as a string pinned a variable to 0.
- `at most 0 of these` compiled to **nothing**. Slack is only allocated when the range has room in
  it, and "needs no slack" was taken to mean "needs no constraint".
- The node-update ceiling counted burn-in only. A request declaring 1,024 node updates did 262
  million, because every handler runs `draws × thin` further sweeps afterwards. `verify` had no
  ceiling at all.
- The node editor discarded every refusal code, so a constraint the library rejected vanished from
  the model and the editor answered a different problem.

### Added

- `ft_model_*`: the whole modelling layer over the C ABI, declared in `include/ferrotherm.h`.
- `ft_model_fixed_penalty`, the remedy every error message recommends and no surface could perform.
- `ft_model_solve_with` and a `schedule` on HTTP/MCP: a caller who measured their instance can say
  so. A ladder that runs backwards is refused rather than substituted.
- `ft_model_violations` / `ft_model_violation`, and `violated` on every surface.
- Certification of a compiled model from Python and Julia, and the five `ft_model_cert_*`
  accessors declared at last.
- `exactly_one` / `at_most_one` on every surface. They lower pairwise with no slack, so they are
  measurably cheaper than `k = 1`.

### Verification

- `include/check.c` compiles against the header and links against the library, because nothing was
  checking that the header describes the ABI. It found a defect on its first run.
- `web-tests/`: 24 browser tests driving `window.ferrotherm`, the same surface an agent uses.
  `npm run live` drives the deployed copy rather than a local build — a stale deploy served a build
  missing a full day of exports while the page still loaded and still answered questions.
- `scripts/check-wasm-exports.sh` derives its requirement from the pages themselves, every
  `W.ft_...` in `docs/*.html`, rather than a list kept by hand beside them.
- The agent harness drives `ferrotherm_solve`, which nothing had, which is how a whole family of
  defects on it lived behind a green suite.

355 Rust, 18 Python, 17 Zig, 117 Julia, 24 browser, 18 C.

## 0.5.1 (2026-08-05)

Deployment-ladder facts finalized at datasheet grade (verified Aug 2026).

- Alchitry V2 lineup: Cu V2 $59.99 (iCE40-HX8K, the full-open-flow rung), Au V2 $149.99
  (XC7A35T-2), Pt V2 $349.99 (XC7A100T-2, 4x GTP = PCIe Gen2 x4 capable; the vendor listings'
  "FGG84I" package is a typo, confirmed FGG484 from the Rev A schematic).
- Kria KV260: XCK26 exact fabric numbers; corrected the widely repeated FALSE claim that it needs
  Vivado Enterprise (free ML Standard covers Kria, per AMD's licensing FAQ).
- Numato Aller: XC7A200T-2 in M.2 2280 — the only first-party 2280 M.2 FPGA still manufactured
  (LiteFury/NiteFury/Acorn dead); ~$500 quote-only.
- AWS f2.48xlarge, architecture-critical [AWS re:Post]: NO FPGA-to-FPGA links (no P2P, no ring —
  F1 had both). The x8 tier therefore runs replica-exchange parallel tempering (scalar energies
  per swap fit host-mediated topology) rather than DSIM-2-style lattice partitioning.

## 0.5.0 (2026-08-05)

The hardware backend and the named deployment ladder.

- `hdl`: lower any bipartite sampling graph to a fixed-point p-bit fabric (Q.8 weights, 1024-entry
  sigmoid ROM, per-node xorshift32, two-phase chromatic schedule) and emit synthesizable Verilog.
  The contract: `FixedFabric` is a CYCLE-EXACT Rust emulator of the emitted RTL — the generated
  self-checking testbench replays the emulator's per-sweep state trace and must match BIT-EXACTLY
  in icarus-verilog simulation (gated in `cargo test`; CI installs iverilog). The quantized
  fabric also re-passes the Onsager physics gate within quantization tolerance. Software model ==
  emulator == RTL, verified.
- `targets`: the named deployment ladder added — Alchitry Au/Au+, AMD Kria KV260 (K26 SOM),
  Numato Aller (XC7A200T in M.2: the compute-stick class, buyable today), and AWS f2.48xlarge
  (8x VU47P, the multi-chip tier for DSIM-2-style distributed Gibbs). 19-entry database.
- CI: icarus-verilog installed so the RTL gate runs on every push.

## 0.4.0 (2026-08-05)

The performance core, the same-machine parity measurement, and the FPGA deployment-target database.

- `gibbs::Sampler::sweep_par` / `sweeps_par`: multithreaded chromatic sweeps (scoped threads,
  race-free by coloring, per-(sweep, class, chunk) RNG streams; bit-reproducible for a fixed
  (seed, threads)). Passes the same Onsager physics gate as the sequential path.
- Same-machine, same-model parity vs THRML (JAX 0.11, Python 3.14, CPU), measured quiet:
  at 16,384 nodes ferrotherm 6.3e7 flips/s single-thread vs THRML 1.68e7 (3.7x); at ~270k nodes
  ferrotherm 3.8e8 at 18 threads vs THRML 1.05e8 (3.6x; THRML's vectorization beats our single
  thread at that size, 9.5 vs 13.6 ns). Browser WebGPU on the same machine: 9.35e9 (89x THRML-CPU;
  THRML has no GPU path on non-CUDA hardware). Scripts: `scripts/thrml_bench.py`,
  `examples/parity_bench.rs`.
- Corrected a published number: the earlier 86 ns/flip CPU figure was measured while background
  jobs shared the machine; quiet re-measurement gives 13.6 ns/flip. Recorded as a discipline rule.
- `targets`: the FPGA deployment-target database — edge parts (iCE40/ECP5/Gowin/Artix, with open-
  toolchain status), buyable cards (Alveo U55C = AWS F2 silicon twin; V80 flagship with a
  first-mover slot), cloud instances (AWS F2 active at $1.98/hr; Azure NP sunsetting May 2027;
  Alibaba/Huawei FPGA clouds verified dead), academic clusters (AMD HACC, NSF OCT: F2-class
  silicon at $0), and a $200 salvage CI tier. Capacity model anchored to the measured DSIM-2
  machine (arXiv:2606.25313: 18x VP1902, 1e12 flips/s) after the anchor test caught the first
  version ~10x optimistic. Large-part and calibration sweeps queued.

## 0.3.0 (2026-08-05)

Wave 3 of the field ingest: the flagship architecture and two more hardware-algorithm lines.

- `dtm`: Denoising Thermodynamic Models (arXiv:2510.23972) — closed-form forward jump kernels,
  pattern grids (G8/G12/G16 with degree and bipartiteness tests), contrastive chain training with
  latent marginalization, the TC penalty (closed form, h-component cancels exactly), the ACP
  controller (scripted-sequence unit test), and reverse-chain sampling. THE GOLD TEST: on a fully
  enumerable DTM the Eq.-14 gradient with exact conditional expectations matches central finite
  differences of the exact NLL for every parameter; sampled-gradient training must reduce the
  EXACT NLL. Recorded: the paper's printed Eq. D1 sign is wrong — the energy-form keep
  probability test pins the negative sign and shows the printed sign yields exactly the
  complement (indistinguishable only in the noise-saturation limit).
- `lrw`: Lattice Random Walk SDE discretisation (arXiv:2508.20883, the algorithm behind Normal
  Computing's CN101) — ternary increments with exact conditional moments (algebraic identity
  test, no Monte Carlo), validity clipping, and the stability mechanism demonstrated: a cubic
  drift that provably diverges under Euler-Maruyama from x0 = 5 stays bounded under the walk.
- `sbm`: Simulated bifurcation (ballistic and discrete variants, Goto et al.) — symplectic
  momentum-first updates, inelastic walls, best-so-far readout. Verified against exhaustively
  enumerated ground states: K8, C7, Petersen exact on both variants; 20/20 and 20/20 on seeded
  N = 16 Gaussian instances. Recorded: with x initialized exactly to zero, symmetric graphs
  synchronize, hit the walls together, and the momentum reset erases the symmetry-breaking
  (measured: dSB converged to the WORST K8 state); a small random x-init removes the trap.

## 0.2.0 (2026-08-05)

Wave 2 of the field ingest. Breaking: `Program::run` gains a trajectory-trace parameter.

- `het`: heterogeneous factor-graph Gibbs — spin and categorical nodes, energy-table factors of
  arbitrary arity (subsumes pairwise), proper-coloring block sweeps, clamping. Verified: a mixed
  spin+categorical model with a 3-ary factor matches exact Boltzmann enumeration (TV < 0.02); on
  pure-spin pairwise models the het and spin engines agree to 1e-12; clamped categorical
  conditionals match exact rows. The spin engine remains the fast path.
- `linalg`: cyclic Jacobi eigensolver for symmetric matrices (reconstruction-verified).
- `tla::solve_spd_exact_ou`: the bias-free exact Ornstein-Uhlenbeck transition integrator in the
  eigenbasis. New tests pin BOTH facts: the Euler-Maruyama chain's stationary covariance is
  biased per eigenmode by exactly 2/(2 - dt*alpha) (the test that catches silently absorbing it),
  and the exact integrator lands on beta^-1 alpha^-1 unbiased.
- `program::Gate::BoltzExact` + `Program::ebm_kernel_grad`: the third gradient estimator
  (EBM-kernel decomposition: one trajectory plus one auxiliary clamped draw, arXiv:2608.01612
  Sec III C). Cross-validated against exact-score REINFORCE and an in-test full-enumeration
  reference. Recorded along the way: finite differences with common random numbers is NOT a
  usable referee across a discrete re-draw (CRN decorrelates; noise floor exceeded the gradient),
  so the test enumerates instead.

## 0.1.0 (2026-08-05)

First public release. Pure Rust, zero dependencies, std-only, wasm-clean, deterministic by seed.

- `graph` + `gibbs`: sparse pairwise energy-based models, chromatic block-Gibbs with clamping.
  Verified: exact Boltzmann on enumerable systems; Onsager's 2D closed form to 4 decimals.
- `device`: the published degree-16 planar topology of Z1-class thermodynamic chips
  (displacement rules (1,0),(2,1),(2,3),(4,1); proven bipartite, longest edge sqrt(17)).
- `ledger`: first-class joules accounting; `Z1_SPICE` prices (pre-silicon vendor estimates,
  arXiv:2608.01615 Table IV) with the write/sample ratio (~21,700x) as a tested invariant.
- `tempering`: simulated annealing + parallel tempering with ladder diagnostics. Verified:
  finds exhaustively-enumerated ground states of frustrated glasses.
- `tla`: thermodynamic linear algebra (Ornstein-Uhlenbeck network; Aifer et al.,
  arXiv:2308.05660). Verified: SPD solves match Gaussian elimination; covariance estimates A^-1.
- `program`: stochastic differentiable circuits (flip, Gaussian-policy, linear-dynamics,
  stage-cost, Gibbs-kernel gates); REINFORCE, parameter-shift, and finite-difference gradients
  cross-validated; a trained stochastic controller reaches a provably optimal LQR gain.
- `compile`: variational compilation of conditional kernels onto device graph patches with
  hidden-unit marginalization; exact positive/negative-phase gradients (FD-verified to 1e-6);
  the chain-rule KL error bound verified exactly on compiled programs.
- `ffi`: C ABI (`ft_*`) for WebAssembly and host languages; FFI path re-verified against Onsager.
- Examples double as verification gates (exit codes); `web/gibbs_bench.html` is the WebGPU
  instrument (verifies against Onsager on the visitor's GPU before reporting throughput).
