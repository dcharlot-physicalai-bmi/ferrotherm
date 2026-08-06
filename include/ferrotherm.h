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
