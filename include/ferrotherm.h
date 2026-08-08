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
uint32_t ft_model_lits(const ft_model *m);   /* how many are pending */

/* kind: 0 exactly k, 1 at most k, 2 at least k, 3 exactly one, 4 at most one.
 * The last two ignore k and lower pairwise, with no slack variable -- prefer them over k = 1. */
uint32_t ft_model_close(ft_model *m, uint32_t kind, uint32_t k);

/* Objective terms. `maximize` is 1 to prefer large, 0 to prefer small. The pair form is quadratic:
 * it rewards two variables taking their values together. */
uint32_t ft_model_objective_term(ft_model *m, uint32_t maximize, double coeff,
                                 uint32_t v, int64_t value);
uint32_t ft_model_objective_pair(ft_model *m, uint32_t maximize, double coeff,
                                 uint32_t a, int64_t av, uint32_t b, int64_t bv);

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

/* How many constraints the answer breaks; zero when it keeps everything it was asked to. Read each
 * with ft_model_violation, which describes it in the caller's own names: "a and b must differ, and
 * both are 1". Same two-call text protocol as the other string getters. */
uint32_t ft_model_violations(const ft_model *m);
uint32_t ft_model_violation(const ft_model *m, uint32_t i, uint8_t *buf, uint32_t cap);

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

/* Release a simulation. */
void ft_free(ft_sim *sim);

#ifdef __cplusplus
}
#endif
#endif /* FERROTHERM_H */
