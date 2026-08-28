/* ferrotherm.h -- C ABI for thermodynamic sampling.
 *
 * One header, no dependencies, no build system. Every language with a C FFI reaches the sampler
 * through this file: Zig via @cImport, Python via ctypes, and anything else the same way.
 *
 * Link against the cdylib built by `cargo build --release` (libferrotherm.so / .dylib / .dll), or
 * load the wasm32-unknown-unknown build in a browser.
 *
 * Conventions
 *   States are -1/+1, held as int8_t.
 *   Energy is  E = -sum_ij J_ij s_i s_j - sum_i h_i s_i.
 *   beta is inverse temperature, 1/T: larger is colder and more ordered.
 *   Handles are opaque and owned by the library. Every function is null-safe.
 *   One simulation is single-threaded; concurrent calls on one handle are the caller's bug.
 *
 * Licence: Apache-2.0. Institute for Physical AI @ BMI.
 */
#ifndef FERROTHERM_H
#define FERROTHERM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handles. */
typedef struct ft_sim ft_sim;
typedef struct ft_builder ft_builder;

/* ---- built-in models ------------------------------------------------------------------------ */

/* 2D nearest-neighbour Ising lattice, periodic, side `l`, coupling `j`. Returns NULL on failure. */
ft_sim *ft_ising2d_new(uint32_t l, double j, double beta, uint64_t seed);

/* Z1-topology grid, degree 16, open boundaries, `w` by `h`, uniform coupling `j` and bias `hb`. */
ft_sim *ft_z1_new(uint32_t w, uint32_t h, double j, double hb, double beta, uint64_t seed);

/* ---- models you define ----------------------------------------------------------------------- */

/* New builder over `n` nodes, or NULL if n is 0. Consume with ft_builder_build, or release with
 * ft_builder_free; doing neither leaks it. */
ft_builder *ft_builder_new(uint32_t n);

/* Add coupling J_ij. Returns 1 on success, 0 if the handle is NULL, an index is out of range,
 * i equals j, or w is not finite. Duplicate pairs are summed at build time. */
uint32_t ft_builder_couple(ft_builder *b, uint32_t i, uint32_t j, double w);

/* Add bias h_i. Returns 1 on success, 0 on a NULL handle, out-of-range index, or non-finite h. */
uint32_t ft_builder_bias(ft_builder *b, uint32_t i, double h);

/* Consume the builder into a simulation. The builder handle is invalid after this call. */
ft_sim *ft_builder_build(ft_builder *b, double beta, uint64_t seed);

/* Release a builder that was never built. */
void ft_builder_free(ft_builder *b);

/* ---- running -------------------------------------------------------------------------------- */

/* Run `n` chromatic block-Gibbs sweeps. Returns total sweeps done so far, or 0 on NULL. */
uint64_t ft_sweep(ft_sim *sim, uint32_t n);

/* Anneal down a geometric ladder from `beta_min` to `beta_max`, leaving the simulation holding the
 * lowest-energy state found and returning that energy. Returns NaN on a NULL handle, on a ladder
 * that is not 0 < beta_min < beta_max, on stages < 2, or on sweeps_per_stage == 0. */
double ft_anneal(ft_sim *sim, double beta_min, double beta_max,
                 uint32_t stages, uint32_t sweeps_per_stage);

/* Change temperature without disturbing the state. */
void ft_set_beta(ft_sim *sim, double beta);

/* ---- reading -------------------------------------------------------------------------------- */

/* Local field at node i: sum_j J_ij s_j + h_i, with beta excluded. NaN on null or out of range.
 * Exposed so a state computed elsewhere can be checked against the field this library computes for
 * it, one node at a time, rather than only by comparing a total. */
double ft_field(const ft_sim *sim, uint32_t i);

/* The graph in the width a GPU actually has: k neighbours per node, padded. ft_gpu_k is that width,
 * ft_gpu_nbr and ft_gpu_w are n*k neighbour indices and f32 couplings, ft_gpu_h is n fields, and
 * ft_gpu_classes / ft_gpu_class_ptr / ft_gpu_class_len describe the colouring that makes a sweep
 * parallel. A caller building device buffers takes them from here so its layout and this library's
 * cannot drift apart. */
uint32_t ft_gpu_k(const ft_sim *sim);
const uint32_t *ft_gpu_nbr(const ft_sim *sim);
const float *ft_gpu_w(const ft_sim *sim);
const float *ft_gpu_h(const ft_sim *sim);
uint32_t ft_gpu_classes(const ft_sim *sim);
const uint32_t *ft_gpu_class_ptr(const ft_sim *sim, uint32_t c);
uint32_t ft_gpu_class_len(const ft_sim *sim, uint32_t c);

/* Put a state INTO the simulation, so something computed elsewhere -- a GPU sweep, another solver --
 * is scored, certified or annealed by exactly the same code that handles a state this library
 * produced. Returns 1 on success, 0 on refusal.
 *
 * It refuses rather than adapting: `len` must equal the node count and every value must be -1 or +1.
 * A short state means whatever produced it did not finish; a value that is not a spin means the
 * buffer is not what the caller thinks it is. Both are cheap to launder into something plausible,
 * and a laundered state is then scored with full confidence -- which is how a dropped GPU dispatch
 * becomes a believable energy. */
uint32_t ft_set_spins(ft_sim *sim, const int8_t *spins, uint32_t len);

/* The WGSL sweep shader, so a caller running this model on its own GPU runs the arithmetic this
 * library tests rather than a second copy of it. NUL-free; pair the pointer with the length. */
const uint8_t *ft_shader(void);
uint32_t ft_shader_len(void);

/* Node count. ft_len and ft_nodes agree; both are provided because callers reach for both names. */
uint32_t ft_len(const ft_sim *sim);
uint32_t ft_nodes(const ft_sim *sim);

/* Pointer to `ft_len` spins, each -1 or +1. The pointer is owned by the library and is valid until
 * the next call that mutates this simulation. Copy before sweeping again. */
const int8_t *ft_spins(const ft_sim *sim);

double ft_energy(const ft_sim *sim);
double ft_magnetization(const ft_sim *sim);

/* ---- energy ledger --------------------------------------------------------------------------- */

/* Node updates charged so far. */
uint64_t ft_ledger_updates(const ft_sim *sim);

/* Those operations priced at Z1-class SPICE figures (arXiv:2608.01615 Table IV). This prices the
 * modelled device, not the CPU actually running the sweep. */
double ft_ledger_joules_z1(const ft_sim *sim);

/* ---- instances with a known optimum ------------------------------------------------------------ */

/* Plant frustrated plaquettes on an `l` by `l` periodic lattice. Each loop contributes -2 to the
 * ground energy, so the optimum is known by construction. NULL if l < 3 or loops == 0.
 *
 * Difficulty is NOT monotonic in `loops`: it peaks near four loops per edge and falls away at both
 * ends, so a very sparse or a saturated instance is easy. */
ft_sim *ft_planted_frustrated(uint32_t l, uint32_t loops, uint64_t seed, double beta);

/* The Wishart planted ensemble: dense, and hard below alpha = 1. A miss here is usually under 2%
 * above the optimum, so report a solve rate rather than a mean excess. */
ft_sim *ft_planted_wishart(uint32_t n, double alpha, uint64_t seed, double beta);

/* The known optimum of a planted instance, or NaN if this simulation is not one. */
double ft_ground_energy(const ft_sim *sim);

/* ---- certificate ------------------------------------------------------------------------------- */

/* Sample `draws` states with `thin` sweeps between them and certify the run. Returns 1 on success,
 * 0 on NULL or fewer than 16 draws. Read the result with the accessors below. */
uint32_t ft_certify(ft_sim *sim, uint32_t draws, uint32_t thin);

double ft_cert_beta_eff(const ft_sim *sim);  /* the temperature actually sampled at */
double ft_cert_beta_lo(const ft_sim *sim);   /* 95% interval, widened for autocorrelation */
double ft_cert_beta_hi(const ft_sim *sim);
double ft_cert_tau(const ft_sim *sim);       /* integrated autocorrelation time */
double ft_cert_ess(const ft_sim *sim);       /* independent samples, not raw draws */
double ft_cert_tv(const ft_sim *sim);        /* distance from exact, where enumerable; else NaN */
double ft_cert_floor(const ft_sim *sim);     /* never quote a distance below this */

/* 1 if the run certified clean. Zero findings is the only thing that means sound. */
uint32_t ft_cert_passed(const ft_sim *sim);
uint32_t ft_cert_findings(const ft_sim *sim);

/* Copy finding `i` into `buf` as UTF-8. Returns bytes written, or the length needed if buf is
 * NULL, or 0 if there is no such finding. */
uint32_t ft_cert_finding(const ft_sim *sim, uint32_t i, uint8_t *buf, uint32_t cap);

/* ---- solvers and bounds -------------------------------------------------------------------------

   Each solver LEAVES ITS BEST STATE as the simulation's state, so ft_spins reads the answer and
   ft_energy recomputes the energy from it rather than trusting the number returned here. That also
   makes them compose: anneal, then tabu from where annealing stopped, then branch and bound with
   that as its incumbent.

   Every bound below is a LOWER bound on the ground energy and is sound on its own, so a caller
   should take the maximum of the ones it can afford. They are ordered cheapest first. */

/* Tabu search. Returns the energy of the best state found, or NaN on NULL.
   tenure = 0 scales the tenure to the graph; restart_after = 0 never restarts. */
double ft_tabu(ft_sim *sim, uint32_t iterations, uint32_t tenure, uint32_t restart_after);

/* Iterations the last ft_tabu actually ran. Less than the budget means the search was truncated,
   which is otherwise invisible from outside. */
uint64_t ft_tabu_iterations(const ft_sim *sim);

/* EXACT max-cut on a planar graph, in polynomial time. Not a search: no budget, no seed, no
   incumbent. Returns the maximum cut weight under w = -J, or NaN when the graph cannot be solved
   this way. The simulation's state becomes the optimal partition, so ft_energy then returns the
   PROVED MINIMUM energy.

   `scale` multiplies every coupling before it is rounded to an integer; pass 1.0 for whole-number
   couplings. The matching underneath is exact only in exact arithmetic, so a weight that does not
   land on an integer is refused rather than rounded. */
double ft_planar_cut(ft_sim *sim, double scale);

/* Faces in the planar embedding -- the dual's vertex count. */
uint64_t ft_planar_faces(const ft_sim *sim);

/* Odd-degree dual vertices: the size of the matching problem and the real cost driver. This is
   what makes the method O(n^3) rather than O(2^n). Zero is a legitimate answer -- a grid with
   uniform weights has every face of even degree and the whole cut comes free. */
uint64_t ft_planar_odd_faces(const ft_sim *sim);

/* Why the last ft_planar_cut refused, in the caller's own terms. Two-call text protocol: pass a
   NULL buffer for the length, then a buffer of that size. Empty on success. There are four
   distinct refusals -- fields, not planar, a cut vertex, non-integral weights -- and they are four
   different things to do next, which a bare NaN collapses into one. */
uint32_t ft_planar_error(const ft_sim *sim, uint8_t *buf, uint32_t cap);

/* An UPPER BOUND on the maximum cut of a toroidal grid, from the same dual reduction. A torus is
   not a plane and ft_planar_cut refuses it; but the dual argument needs only faces, and an
   embedding on any surface has them. On a torus the cycle space of the dual is four times the cut
   space, so the relaxation ranges over sets that are not cuts and its optimum bounds the maximum
   from above.

   That is the side of G-set nobody publishes: every figure in the table is a best cut FOUND, a
   lower bound. Measured, this closes the bracket on G11, proving its best-known cut of 564 optimal.

   Returns NaN unless the graph is a toroidal grid, whose structure is recovered from the edge list
   -- a match on all 2n edges rather than a guess. */
double ft_toroidal_bound(ft_sim *sim, double scale);

/* 1 if the last ft_toroidal_bound was ATTAINED by a genuine cut, in which case it is the maximum
   rather than a bound, and the simulation's state is the partition achieving it. Not attained still
   leaves the bound standing: every cut is such a subgraph, so a maximum over the larger set can
   only be larger. */
uint32_t ft_toroidal_attained(const ft_sim *sim);

/* Goemans-Williamson: round the semidefinite relaxation to a state. THE ONLY WORST-CASE GUARANTEE
   IN MAX-CUT. ft_bound_sdp uses the relaxation from the dual side for a bound; this uses it from
   the primal side for a solution. Returns the cut under w = -J and leaves the state on the sim.

   The 0.87856 ratio DOES NOT APPLY IN GENERAL -- it is stated for non-negative edge weights, which
   here means non-positive couplings and no fields. Ask ft_gw_guaranteed, because a guarantee that
   is always claimed is not a guarantee. */
double ft_gw_round(ft_sim *sim, uint32_t hyperplanes, uint64_t seed);
uint32_t ft_gw_guaranteed(const ft_sim *sim);

/* Parallel tempering with ISOENERGETIC CLUSTER MOVES -- the baseline the field measures against.
   Two ladders of `rungs` replicas; every round a connected component of the disagreement subgraph
   between the two replicas at each temperature is flipped in BOTH. The move preserves the pair's
   energy exactly and is therefore always accepted, which is what makes it a cluster algorithm for
   a spin glass.

   NaN when the graph carries a FIELD: the isoenergetic argument holds only at h = 0, and accepting
   the move anyway would be silently wrong. */
double ft_icm(ft_sim *sim, uint32_t rungs, uint32_t rounds, double beta_min, double beta_max);

/* Cluster moves that actually fired in the last ft_icm. A move that never fires is not a move. */
uint64_t ft_icm_moves(const ft_sim *sim);

/* Simulated quantum annealing: path-integral Monte Carlo on the transverse-field Ising model.
   `trotter` slices at fixed beta, transverse field annealed from gamma_max to gamma_min over
   `steps`. ONE SLICE IS CLASSICAL, which is the honest control rather than a degenerate case.
   gamma_min is clamped away from zero, where the Trotter coupling diverges. */
double ft_sqa(ft_sim *sim, uint32_t trotter, double beta, double gamma_max, double gamma_min,
              uint32_t steps);

/* Breakout local search -- the algorithm that holds the max-cut record on most of G-set. Steepest
   descent with an adaptive perturbation between local optima. Returns the best energy found, or NaN
   on NULL. One iteration is one SPIN FLIP, which is what ft_tabu counts too, so passing the same
   number to both is a matched-budget comparison. */
double ft_bls(ft_sim *sim, uint32_t iterations);

/* Local optima the last ft_bls visited. A run with a handful of descents spent its budget inside
   one basin and is a descent, not a breakout search. */
uint64_t ft_bls_descents(const ft_sim *sim);

/* Flips the last ft_bls actually made, which is not always the budget it was given. */
uint64_t ft_bls_iterations(const ft_sim *sim);

/* The largest jump magnitude the last ft_bls reached. It grows only when a descent returns to the
   immediately previous local optimum, so a value above the initial L0 is evidence the adaptive rule
   fired rather than idled. */
uint32_t ft_bls_max_jump(const ft_sim *sim);

/* Population annealing on a linear ladder from beta = 0 to beta_max in `stages` steps. Returns the
   best energy found, or NaN on NULL or a bad beta_max. Starting at zero, where Z = 2^n exactly, is
   what makes ft_popanneal_ln_z an absolute free energy rather than a ratio. */
double ft_popanneal(ft_sim *sim, uint32_t population, uint32_t sweeps, double beta_max,
                    uint32_t stages);

/* ln Z at the final beta from the last ft_popanneal, or NaN if there was none. */
double ft_popanneal_ln_z(const ft_sim *sim);

/* The worst family statistic rho over the ladder -- THE NUMBER THAT SAYS WHETHER TO BELIEVE
   ft_popanneal_ln_z. 1.0 means every ancestor still has one descendant; the population size means
   the population collapsed onto one ancestor and explored a single basin with N copies of one
   history. NaN if no run has happened. */
double ft_popanneal_rho(const ft_sim *sim);

/* Branch and bound, starting from this simulation's current state as its incumbent. Returns the
   lowest energy found. WHETHER IT IS THE MINIMUM IS A SEPARATE QUESTION -- ask ft_branch_proved. */
double ft_branch(ft_sim *sim, uint64_t max_nodes);

/* 1 if the last ft_branch exhausted the tree and its answer is the proved minimum, else 0. A run
   that ran out of nodes returns the best state it saw and reports 0 here. */
uint32_t ft_branch_proved(const ft_sim *sim);
uint64_t ft_branch_nodes(const ft_sim *sim);

/* min E >= -sum|h| - sum|J|, in O(edges). The cheapest bound and the weakest. */
double ft_bound_decoupled(const ft_sim *sim);

/* Lagrangian decomposition into forests, tightened by `rounds` of subgradient ascent. WORTH
   NOTHING ON AN INSTANCE WITH NO FIELDS: a tree is never frustrated, so every part minimises to
   -sum|J| and this degenerates to ft_bound_decoupled. */
double ft_bound_forest(const ft_sim *sim, uint32_t rounds);

/* Charges 2*min|J| for every edge-disjoint frustrated cycle up to length max_len. Edge-disjointness
   is what makes the penalties add. */
double ft_bound_odd_cycle(const ft_sim *sim, uint32_t max_len);

/* The certified semidefinite bound, RE-VERIFIED AT THIS BOUNDARY before it is returned: the cost
   matrix is rebuilt from the graph and the positive-definiteness proof re-run, and NaN comes back
   if that fails. A bound crossing a language boundary is exactly the case where the caller cannot
   check it themselves. */
double ft_bound_sdp(const ft_sim *sim, uint32_t sweeps, uint64_t seed);

/* ---- exact inference --------------------------------------------------------------------------- */

/* Exact ground energy by variable elimination, or NaN if the induced width exceeds `max_width`.
 * Cost is 2^width in the graph's SHAPE, not 2^n in its size: a long chain is instant where a dense
 * graph of the same node count is impossible. Check ft_exact_width first. */
double ft_exact_ground(const ft_sim *sim, uint32_t max_width);

/* Exact log partition function at `beta`, or NaN if too wide. */
double ft_exact_log_z(const ft_sim *sim, double beta, uint32_t max_width);

/* Exact ground STATE by variable elimination, written into `out` as -1/+1. Returns 1 on success,
 * 0 on NULL, a wrong length, or a graph wider than max_width. A caller that must return a solution
 * rather than a bound needs this, not just the energy. */
uint32_t ft_exact_ground_state(const ft_sim *sim, uint32_t max_width, int8_t *out, uint32_t len);

/* Induced width of the elimination order. */
uint32_t ft_exact_width(const ft_sim *sim);

/* ---- modelling ------------------------------------------------------------------------------- */
/*
 * Everything above works in spins. This works in problems: variables that hold values, constraints
 * that say what must be true, an objective that says what is better. It compiles to the layer above
 * and answers in the values the caller declared.
 *
 * Values are int64_t and mean what the modeller wrote. An integer variable over 10..=20 holds the
 * value 13 in its fourth slot, and every function here takes 13. Passing 3 is an error naming the
 * range, not an answer to a different question.
 */

typedef struct ft_model ft_model;

/* A new, empty model. Free it with ft_model_free. */
ft_model *ft_model_new(void);
void ft_model_free(ft_model *m);

/* Declare a variable, returning its index, or UINT32_MAX if the domain is unusable. */
uint32_t ft_model_categorical(ft_model *m, uint32_t values); /* needs at least 2 */
uint32_t ft_model_integer(ft_model *m, int64_t lo, int64_t hi); /* needs hi > lo */
uint32_t ft_model_binary(ft_model *m);

/* Give a variable the caller's own name, so errors and reports use it. Optional: an unnamed
 * variable is called v0, v1 and so on. `len` is a byte count; the bytes need no terminator. */
uint32_t ft_model_name(ft_model *m, uint32_t v, const uint8_t *name, uint32_t len);

/* Declare a variable with a chosen ENCODING: 0 one-hot, 1 binary, 2 domain-wall.
 *
 * The trade is the difference between a model that fits a machine and one that does not:
 *
 *   encoding      spins for k values     usable in a constraint or objective
 *   one-hot       k                      yes
 *   domain-wall   k - 1                  yes
 *   binary        ceil(log2 k)           NO
 *
 * Only a one-hot or domain-wall indicator has a bounded degree in the spins. A binary code's
 * indicator is a product of every bit, so its degree grows with the domain; such a variable is
 * cheapest to store and is refused BY NAME if it appears in a literal, rather than expanded into
 * something nobody wants to read.
 *
 * Domain-wall is often the better choice for an INTEGER, which is an ordered domain: neighbouring
 * values sit one spin flip apart where one-hot puts them two apart, and it costs one spin fewer.
 * It is not the default only because one-hot is the safe answer for a caller who has not thought
 * about it. Returns the variable index, or UINT32_MAX. */
uint32_t ft_model_categorical_as(ft_model *m, uint32_t values, uint32_t encoding);
uint32_t ft_model_integer_as(ft_model *m, int64_t lo, int64_t hi, uint32_t encoding);

/* Constraints. Each returns 1 on success, 0 on refusal; ft_model_error says why. */
uint32_t ft_model_not_equal(ft_model *m, uint32_t a, uint32_t b);
uint32_t ft_model_equal(ft_model *m, uint32_t a, uint32_t b);
uint32_t ft_model_fix(ft_model *m, uint32_t v, int64_t value);

/* Counting constraints over two to four variables, each taking `value`. Pass UINT32_MAX for the
 * unused slots and set `count` to how many are real.
 *
 * The three differ only in the comparison, and the difference is not cosmetic: an equality can be
 * squared directly, while an inequality needs a slack variable to become one. at_most and at_least
 * therefore cost extra spins. The slack never appears in the answer. */
uint32_t ft_model_cardinality(ft_model *m, uint32_t count, uint32_t k, int64_t value,
                              uint32_t a, uint32_t b, uint32_t c, uint32_t d);
uint32_t ft_model_at_most(ft_model *m, uint32_t count, uint32_t k, int64_t value,
                          uint32_t a, uint32_t b, uint32_t c, uint32_t d);
uint32_t ft_model_at_least(ft_model *m, uint32_t count, uint32_t k, int64_t value,
                           uint32_t a, uint32_t b, uint32_t c, uint32_t d);

/* Counting constraints of ANY length, whose literals may each name a different value.
 *
 * The positional forms above take four variables and one shared value, which is what a node graph
 * with a fixed number of ports needs and what a scheduling problem does not: "at most two of these
 * nine shifts" cannot be said that way at all. Build a list with ft_model_lit, then close it.
 *
 * The list lives on the model. ft_model_close clears it whether it succeeds or not, so a refused
 * constraint cannot bleed into the next one. */
uint32_t ft_model_lits_clear(ft_model *m);
uint32_t ft_model_lit(ft_model *m, uint32_t var, int64_t value);
/* Append a VARIABLE, for constraints that are about variables rather than literals -- all_different
 * (kind 5) is the only one today. The library picks a value from the variable's own domain, because
 * a caller has no reason to know one and a placeholder passed through ft_model_lit is refused for
 * any variable whose domain does not contain it. */
uint32_t ft_model_var(ft_model *m, uint32_t var);
uint32_t ft_model_lits(const ft_model *m);   /* how many are pending */

/* kind: 0 exactly k, 1 at most k, 2 at least k, 3 exactly one, 4 at most one.
 * The last two ignore k and lower pairwise, with no slack variable -- prefer them over k = 1.
 *
 * kind 5 is ALL-DIFFERENT: it reads the VARIABLES out of the pending literals, ignores their
 * values, and constrains every one of them to take a different value. Write ft_model_lit(m, v, 0)
 * once per variable and close with kind 5; k is ignored.
 *
 * It lowers per shared value rather than per pair, so it costs nothing where two domains do not
 * overlap, needs no slack and no ancillas, and its violation names WHICH value collided and who
 * took it. More variables than the values they share is refused at compile time by name -- the
 * pigeonhole principle, checked rather than annealed, because a model with no answer returns
 * infeasible for a reason no penalty and no longer ladder will fix. */
uint32_t ft_model_close(ft_model *m, uint32_t kind, uint32_t k);

/* SOFT constraints: a preference with a price, rather than a rule.
 *
 * A hard constraint says which answers are answers at all, so breaking one makes ft_model_feasible
 * zero. A soft one is a preference the solver may trade away: breaking it costs, and the answer
 * stays feasible. Both compile to the same squared penalty; what differs is what the answer means.
 *
 * ft_model_close_soft closes the pending literal list as a soft counting constraint (same kind
 * codes as ft_model_close). ft_model_soften_last makes the constraint added most recently soft,
 * for the pairwise ones that take their arguments directly.
 *
 * The weight is ABSOLUTE, not scaled. Automatic scaling exists to stop a hard constraint being
 * outbid by the objective; a soft one is meant to be traded against it.
 *
 * The price is weight * amount SQUARED, and the square is not a detail: a constraint becomes an
 * energy term by squaring how far outside it sits, so missing by two costs four times missing by
 * one. Pricing a preference chooses that curve as well as its scale. */
uint32_t ft_model_close_soft(ft_model *m, uint32_t kind, uint32_t k, double weight);
uint32_t ft_model_soften_last(ft_model *m, double weight);

/* What the broken soft constraints cost. Zero when none broke, or before solving. */
double ft_model_soft_cost(const ft_model *m);

/* 1 if violation i is a hard one, 0 if it is a preference that was traded away. */
uint32_t ft_model_violation_is_hard(const ft_model *m, uint32_t i);

/* Read an ommx.v1.Instance and return a simulation over it, or NULL if it cannot be read.
 *
 * The direction that makes this a bridge rather than an exporter: a problem someone else compiled
 * to OMMX becomes something this sampler can run.
 *
 * constant_out, when non-NULL, receives the offset the 0/1 to -1/+1 substitution introduces:
 *
 *     ommx_objective(x) == ft_energy(sim) + constant
 *
 * On NULL, ft_ommx_error says why: a continuous variable, a bound that is not [0,1], an objective of
 * degree three or more. This sampler samples spins, and a bridge that silently dropped what it could
 * not represent would hand back a model that solves a different problem. */
ft_sim *ft_ommx_read(const uint8_t *bytes, uint32_t len, double beta, uint64_t seed,
                     double *constant_out);
uint32_t ft_ommx_error(uint8_t *buf, uint32_t cap);

/* Serialise the compiled model as an ommx.v1.Instance -- OMMX being the interchange format this
 * corner of the field converged on, so a ferrotherm program can be read by everyone else's tools.
 *
 * Same two-call protocol as the text getters, except the payload is BINARY protobuf rather than
 * UTF-8: call with a NULL buffer for the length, then again with a buffer that size.
 *
 * The objective needs no correction. ferrotherm's spins are -1/+1 and OMMX binaries are 0/1, and the
 * substitution introduces an offset -- which the exporter APPLIES, writing it into the instance:
 *
 *     ommx_objective(x) == ferrotherm_energy(s),   s_i = 2*x_i - 1
 *
 * ft_model_ommx_constant reports that offset so the substitution is visible. Do NOT add it to the
 * objective; it is already there, and adding it again is wrong by exactly its own value. */
uint32_t ft_model_ommx(const ft_model *m, uint8_t *buf, uint32_t cap);
double ft_model_ommx_constant(const ft_model *m);

/* Compile-time CAVEATS: what the compiler knows is wrong with the model and cannot fix.
 *
 * Today there is one kind: an encoding no penalty can make exact. A binary encoding of k values
 * uses ceil(log2 k) spins, spelling 2^ceil(log2 k) codewords; when k is not a power of two the
 * spare codewords decode to nothing and no pairwise penalty separates them from the valid ones.
 * Measured on k = 6: the cheapest INVALID state costs exactly what the cheapest valid one does, so
 * the sampler has no reason to prefer an answer, and ft_model_value reports "did not decode".
 *
 * Read these after ft_model_compile and before trusting a result. Zero is the normal case. */
uint32_t ft_model_caveats(const ft_model *m);
uint32_t ft_model_caveat(const ft_model *m, uint32_t i, uint8_t *buf, uint32_t cap);

/* Objective terms. `maximize` is 1 to prefer large, 0 to prefer small. The pair form is quadratic:
 * it rewards two variables taking their values together. */
uint32_t ft_model_objective_term(ft_model *m, uint32_t maximize, double coeff,
                                 uint32_t v, int64_t value);
uint32_t ft_model_objective_pair(ft_model *m, uint32_t maximize, double coeff,
                                 uint32_t a, int64_t av, uint32_t b, int64_t bv);

/* Add coeff * l1 * l2 * ... * lk to the objective, over the pending literal list.
 *
 * Build the list with ft_model_lit exactly as for a counting constraint, then close it here instead
 * of with ft_model_close. The list is cleared either way, so a refused term cannot bleed into the
 * next one.
 *
 * Three or more literals is a HIGHER-ORDER term. ft_model_compile lowers it by introducing an
 * ancilla spin per substituted pair, so the spin count it returns exceeds what the declared
 * variables need. The guarantee that comes with that lowering is about OPTIMISATION: ground states
 * correspond exactly, and the Boltzmann distribution over the original variables does not survive
 * at finite temperature.
 *
 * One literal is an ordinary linear term and two is ft_model_objective_pair; both are accepted here
 * so a caller building terms in a loop needs one code path rather than three. */
uint32_t ft_model_objective_product(ft_model *m, uint32_t maximize, double coeff);

/* Compile to spins, returning how many were needed, or 0 on failure (see ft_model_error).
 * The count includes any slack an inequality required. */
uint32_t ft_model_compile(ft_model *m);

/* Solve, keeping the best of `tries` anneals. Returns 1 on success. */
uint32_t ft_model_solve(ft_model *m, uint32_t tries);

/* Solve on a caller's own annealing ladder: beta0 to beta1 over `stages` of `sweeps` each.
 * Zero for any of the four means "use the default", so a caller can override only what they
 * measured. Returns 0 if the ladder runs backwards or is not a real number, rather than quietly
 * substituting the default and answering a question nobody asked. */
uint32_t ft_model_solve_with(ft_model *m, uint32_t tries, double beta0, double beta1,
                             uint32_t stages, uint32_t sweeps);

/* The solved value of a variable, in its own units, or INT64_MIN if it did not decode. */
int64_t ft_model_value(const ft_model *m, uint32_t v);

/* 1 when every variable decoded AND every constraint holds.
 *
 * Both halves matter and they fail differently. A variable that did not decode cannot be read at
 * all -- its spins are not a valid codeword. A violated constraint means every value read cleanly
 * and one of them is not what was asked for, which nothing in the values themselves reveals. A
 * penalty makes a constraint expensive, not impossible, so a sampler whose objective outbids it
 * will return exactly that. */
uint32_t ft_model_feasible(const ft_model *m);

/* Spins the higher-order lowering added, or 0 if no objective term named three or more variables.
 * Zero after a failed compile as well, so read it beside a non-zero ft_model_compile. */
uint32_t ft_model_ancillas(const ft_model *m);

/* How many constraints the answer breaks; zero when it keeps everything it was asked to. Read each
 * with ft_model_violation, which describes it in the caller's own names: "a and b must differ, and
 * both are 1". Same two-call text protocol as the other string getters. */
uint32_t ft_model_violations(const ft_model *m);
uint32_t ft_model_violation(const ft_model *m, uint32_t i, uint8_t *buf, uint32_t cap);

/* How far outside constraint `i` the answer sits, in that constraint's own units: places over a
 * ceiling, places under a floor, distance from a fixed value. Always positive; NaN if there is no
 * violation i. The description says a constraint broke; this says whether it was a near miss or a
 * rout, which is what a caller ranking repairs actually needs. */
double ft_model_violation_amount(const ft_model *m, uint32_t i);

double ft_model_energy(const ft_model *m);

/* The penalty weight actually used, which is raised automatically above the largest objective
 * coefficient so that a constraint cannot be outbid. */
double ft_model_penalty(const ft_model *m);

/* Use exactly this penalty, disabling the automatic scaling. This is the remedy when
 * ft_model_feasible returns 0: a constraint lost to the objective and needs to outrank it. Refuses
 * anything that is not a positive number. */
uint32_t ft_model_fixed_penalty(ft_model *m, double p);

/* Two-call text protocol: call with buf NULL and cap 0 for the length, then again with a buffer.
 * Neither writes a terminator; the return value is the byte count. */
uint32_t ft_model_error(const ft_model *m, uint8_t *buf, uint32_t cap);
uint32_t ft_model_ftp(const ft_model *m, uint8_t *buf, uint32_t cap);

/* Certify the compiled model's sampling, the same instrument the raw graphs use. An answer says
 * WHAT; a certificate says whether the machine that produced it was sampling the distribution it
 * claimed. Read findings first: empty is the only value that means the run was sound. */
uint32_t ft_model_certify(ft_model *m, double beta, uint32_t draws, uint32_t thin);
uint32_t ft_model_cert_findings(const ft_model *m);
uint32_t ft_model_cert_finding(const ft_model *m, uint32_t i, uint8_t *buf, uint32_t cap);

/* What the certificate measured. NaN before a certify call, and NaN for tv/floor on a model too
 * large to enumerate exactly. Compare tv against floor, never against zero. */
double ft_model_cert_beta(const ft_model *m);  /* the inverse temperature actually reached */
double ft_model_cert_ess(const ft_model *m);   /* effective sample size */
double ft_model_cert_tau(const ft_model *m);   /* integrated autocorrelation time */
double ft_model_cert_tv(const ft_model *m);    /* total variation from the exact distribution */
double ft_model_cert_floor(const ft_model *m); /* the sampling noise floor tv must beat */

/* ---- reference ------------------------------------------------------------------------------- */

/* Onsager's exact spontaneous magnetisation for the 2D Ising model at inverse temperature `beta`.
 * Ground truth to check a lattice simulation against; 0 above the critical point. */
double ft_onsager(double beta);

/* ---- higher-order models ---------------------------------------------------------------------- */
/*
 * Everything above is pairwise, or becomes pairwise. A k-body term CAN be expressed through
 * ft_model_objective_product, which quadratises it with ancillas -- but that is a different
 * computation with a measured cost: examples/hubo_vs_reduction gives the reduced path its best
 * ladder and 1024x the budget and it still does not reach the native path at 1x, because the
 * reduction's penalty is ~1300 against term weights of 1 and the landscape goes rigid.
 *
 * This is the native path. A term of any width contributes -w * prod(s_i), no ancillas anywhere.
 *
 * Refusal convention throughout: uint32_t functions return 1 on success and 0 on refusal, double
 * functions return NaN, constructors return NULL. ft_hubo_error carries the reason.
 */
typedef struct ft_hubo ft_hubo;

/* A model over `n` spins, or NULL if n is 0. Free with ft_hubo_free. */
ft_hubo *ft_hubo_new(uint32_t n);
void ft_hubo_free(ft_hubo *h);

/* Lift a pairwise simulation into a higher-order model unchanged, so both paths can score the
 * same state. NULL on a null sim. */
ft_hubo *ft_hubo_from_sim(const ft_sim *sim);

/* Build a term of ANY arity: clear, push each variable, close with a weight. ft_hubo_var refuses a
 * variable out of range or already pending (s*s = 1 would silently change the term's order), and
 * ft_hubo_add clears the pending list whether it succeeds or not, so a refused term cannot be
 * absorbed by the next one. */
uint32_t ft_hubo_vars_clear(ft_hubo *h);
uint32_t ft_hubo_var(ft_hubo *h, uint32_t var);
uint32_t ft_hubo_vars(const ft_hubo *h);
uint32_t ft_hubo_add(ft_hubo *h, double weight);

/* One to four variables positionally, for a node graph with a fixed number of ports. `count` says
 * how many of a b c d to read; UINT32_MAX marks an unused slot. Anything wider goes through
 * ft_hubo_var + ft_hubo_add, which has no arity ceiling. */
uint32_t ft_hubo_term(ft_hubo *h, uint32_t count, double weight,
                      uint32_t a, uint32_t b, uint32_t c, uint32_t d);

/* The model's own numbers. ft_hubo_ancillas_avoided is an UPPER BOUND on what a pairwise reduction
 * would have spent, not the cost: reduce shares one ancilla across every term containing the same
 * pair, so on three terms sharing one it spends one where this returns three. */
uint32_t ft_hubo_len(const ft_hubo *h);
uint32_t ft_hubo_terms(const ft_hubo *h);
uint32_t ft_hubo_max_arity(const ft_hubo *h);
uint32_t ft_hubo_ancillas_avoided(const ft_hubo *h);

/* Anneal, returning the best energy or NaN on a refusal. Zero for any ladder parameter means "use
 * the default for that one"; NaN is refused rather than read as a zero. */
double ft_hubo_anneal(ft_hubo *h, double beta_min, double beta_max,
                      uint32_t stages, uint32_t sweeps_per_stage, uint64_t seed);

/* Read the state three ways. ft_hubo_spins is valid until the next ft_hubo_* call on this handle;
 * ft_hubo_read refuses a length that is not exactly the model's and never writes partially;
 * ft_hubo_set_spins refuses any element that is not -1 or +1, and refuses the whole write. */
const int8_t *ft_hubo_spins(const ft_hubo *h);
uint32_t ft_hubo_read(const ft_hubo *h, int8_t *out, uint32_t len);
uint32_t ft_hubo_set_spins(ft_hubo *h, const int8_t *ptr, uint32_t len);

/* Score the current state, and probe one flip of it. ft_hubo_delta is the higher-order twin of
 * ft_field: what lets another language or a GPU check this library's arithmetic term by term. */
double ft_hubo_energy(const ft_hubo *h);
double ft_hubo_delta(const ft_hubo *h, uint32_t i);

/* What the last run did. Without the counters, a run that flipped nothing looks like a completed
 * one. Joules are what this WOULD have cost on a Z1-class device (vendor SPICE, pre-silicon). */
uint64_t ft_hubo_proposals(const ft_hubo *h);
uint64_t ft_hubo_accepted(const ft_hubo *h);
double ft_hubo_joules_z1(const ft_hubo *h);

/* The last refusal as UTF-8. Two-call protocol: pass a NULL buf for the length, then a buffer. */
uint32_t ft_hubo_error(const ft_hubo *h, uint8_t *buf, uint32_t cap);

/* Sweep across `threads` OS threads, returning total sweeps done -- same contract as ft_sweep.
 *
 * Within a colour class no two nodes are adjacent, so the class splits into disjoint chunks and
 * each thread reads other-colour spins nobody is writing. Bit-reproducible for a fixed
 * (seed, threads): a DIFFERENT thread count is a different, equally valid sample path, so record
 * the thread count beside the seed or the run is not reproducible from what you wrote down.
 *
 * `threads` of 0 means "ask the machine", which is ft_hardware_threads(). A browser has no threads
 * to spread across and runs serially whatever is asked -- ft_threads_used says what actually ran. */
uint64_t ft_sweep_par(ft_sim *sim, uint32_t n, uint32_t threads);

/* How many threads the last ft_sweep_par ACTUALLY used, or 0 before one. Not the number passed in:
 * a browser answers 1, and a colour class of three nodes cannot occupy eight workers. */
uint32_t ft_threads_used(const ft_sim *sim);

/* How many threads this machine can run at once, or 1 when that cannot be known (a browser).
 * An 18-core machine sampling on one core is the commonest way this library is left slow, and this
 * is the only place a C caller can learn the number without guessing. */
uint32_t ft_hardware_threads(void);

/* Release a simulation. */
void ft_free(ft_sim *sim);

#ifdef __cplusplus
}
#endif
#endif /* FERROTHERM_H */
