#!/usr/bin/env bash
#
# The mutations, written down and run every time.
#
# `mutation-check.sh` breaks one line and asks whether a named test notices. It works, and it takes
# five arguments, so every mutation ever run through it was typed at a prompt once and then lost.
# That means "these tests have teeth" was established at a moment and never re-established, while
# the code underneath moved for months.
#
# This is the same tool with the mutations recorded. Each row is an invariant this project has
# actually got wrong, the smallest edit that reintroduces the error, and the test that must go red.
#
# The gap it exists to close was visible this week. The LP bound parser handed every one-sided
# `x >= lo` an invented upper bound of `lo + 1`, so `Maximize t` subject to `t >= 10` answered 11 --
# a confident optimum to an unbounded problem. 320 tests were green. They were green because every
# bound test used the two-sided form, so no test could distinguish the right answer from the wrong
# one. A suite is only evidence about the cases it can tell apart, and the only way to find out
# which those are is to break the code and watch.
#
# A row that DOES NOT APPLY is a failure, not a skip: the pattern drifted, so the mutation silently
# stopped testing anything and the row has been reporting a pass over nothing.
#
#   scripts/mutation-suite.sh
#
# Restores from git, so it refuses to run on a dirty tree.

set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

# A STALE CRASH SENTINEL MEANS THIS TREE IS NOT TRUSTWORTHY, and it is checked BEFORE the dirty
# check below -- because a leftover mutant makes the tree dirty, so the dirty check always wins the
# race and answers "commit first", which is the opposite of what happened and would have the reader
# committing a deliberate defect. (I wrote it in the other order first and watched it do exactly
# that.) `mutation-check.sh` recovers a sentinel when it runs a row; this precheck is one of the
# conditions this project gates a PUSH on and exits before any row runs, so it must say so itself.
# See that script for why a sentinel exists at all: a trap cannot catch SIGKILL.
if [[ -f "$here/.mutation-in-flight" ]]; then
  echo "REFUSING: .mutation-in-flight says a previous run was killed mid-mutation with" >&2
  echo "          '$(cat "$here/.mutation-in-flight")' left mutated, and never restored it." >&2
  echo "          Run scripts/mutation-check.sh (it recovers on startup), or restore that file." >&2
  exit 2
fi

if ! git diff --quiet; then
  echo "refusing to run: the tree is dirty, and every mutation restores with git checkout." >&2
  echo "commit first — that is the whole safety property." >&2
  exit 2
fi

# file | old | new | test filter | label | package (optional; omit for the root crate)
#
# Keep `old` long enough to be unambiguous and short enough to survive unrelated edits nearby.
mutations=(
  # The defect that forced 0.12.0. Put the invented upper bound back and the new test must catch it.
  "src/lp.rs|[name, \">=\", _] => err(|[name, \">=\", lo] => return Ok((num(lo)?, name.to_string(), num(lo)? + 1)), #[allow(unreachable_patterns)] [name, \">=\", _] => err(|a_bound_with_no_upper_end_is_refused_rather_than_invented|an unbounded LP bound"

  # Determinism. HashMap iteration order decides CSR neighbour order, which decides the order every
  # local field is summed in, which changes the last bits of the energy. Five runs, five programs.
  "src/graph.rs|BTreeMap<(u32, u32), f64> = std::collections::BTreeMap::new()|HashMap<(u32, u32), f64> = std::collections::HashMap::new()|a_graph_builds_bit_identically_every_time|CSR order via HashMap"

  # A NaN objective coefficient compiled, solved, and reported feasible while silently disabling
  # every other preference -- comparisons against NaN are all false, so the sampler stopped
  # improving. Remove the guard and the refusal test must notice.
  # The filter here was the bare substring `not_finite` until wave 3 added
  # `cuts::tests::a_graph_too_large_or_not_finite_is_refused_by_name`, at which point it named TWO
  # tests and this row quietly stopped being evidence about the one it names -- a mutation in
  # `model` would still have gone red, and the suite would still have printed "caught". Nothing in
  # `cuts` has anything to do with `model`: A SUBSTRING FILTER IS A CLAIM ABOUT THE WHOLE TEST
  # LIST, so any module added anywhere can hollow out any row. `check_filters` caught it before a
  # single mutation ran, which is the second time that gate has paid for itself.
  "src/model.rs|self.objective.check_finite(\"objective coefficient\")?;|let _ = self.objective.check_finite(\"objective coefficient\");|a_coefficient_that_is_not_finite_is_refused|NaN objective coefficient"

  # scale_to_fit bounded the ANSWER instead of the work: above a 1e6 candidate ceiling it refused
  # without trying the largest candidate, which is usually the answer.
  # Reintroduced through the loop guard rather than the `if`, because the original defect was
  # written `|| top > 1e6` and a literal `|` is this table's field separator.
  "src/fabric.rs|while n >= 1.0 && tried < 1_000_000 {|while n >= 1.0 && tried < 1_000_000 && top <= 1e6 {|wide_integral|scale_to_fit ceiling refusal"

  # `Sum for f64` folds from -0.0, so a model with no soft violations reported \"-0\".
  "src/model.rs|.sum::<f64>() + 0.0|.sum::<f64>()|a_soft_constraint_is_a_price_not_a_rule|soft cost negative zero"

  # A `Device::run` that accepts a seed and swallows it reports reproducibility it does not have:
  # the caller varies the seed, gets one answer every time, and reads a deaf sampler as a confident
  # one. `conform`'s determinism case CANNOT catch this -- an ignored seed is perfectly
  # reproducible -- so the tests that can are the ones named here.
  "gpu/src/lib.rs|let offset = (seed as u32).wrapping_mul(0x9E37_79B9);|let offset = 0u32; let _ = seed;|the_seed_selects_a_stream_rather_than_being_swallowed|Device::run swallows its seed|ferrotherm-gpu"

  # A BUSY MACHINE HAS NO IDLE. `Meter::idle` guarded the DELTA against the baseline's noise and
  # never checked the baseline was idle, so a complete energy table was published here from a
  # machine at load 82 -- other sessions' work charged to this workload. The contamination inflates
  # the baseline, which overstates the idle share, which is the direction that flattered the very
  # argument being made. Leave the guard in place but make it always permit, and the test that
  # exercises the DECISION (rather than whatever this machine happens to be doing) must go red.
  "meter/src/lib.rs|Some(l) if l > QUIET_LOAD => Err(format!(|Some(l) if false && l > QUIET_LOAD => Err(format!(|the_quiet_threshold_refuses_a_loaded_machine_and_names_the_reason|idle baseline on a busy machine|ferrotherm-meter"

  # SOUNDNESS of the optimality bound. Drop one edge from the forest partition and `E = sum of the
  # parts` stops holding, so the "bound" describes a different problem -- still a number, still
  # plausible, and free to sit ABOVE the true minimum, which reports a NEGATIVE gap and makes every
  # conclusion drawn from it backwards. Checked against brute force on 200 random instances.
  "src/bound.rs|                forest.push((i, j, w));|                if !(i == 0 && j == 1) { forest.push((i, j, w)); }|the_partition_covers_every_edge_exactly_once|optimality bound drops an edge"

  # Subgradient ascent is not monotone -- 145 of 200 random instances peak before the last round --
  # so the bound must be the best round seen, not the final one. `if true` takes the last. This
  # mutation SURVIVED the whole suite on its first outing, because `forest(g, r)` already maximises
  # over rounds 0..r and no test could see inside that; `Bound::best_round` exists to make the
  # difference observable from outside.
  "src/bound.rs|        if total > best {|        if true {|the_bound_is_the_best_round_not_the_last_one|optimality bound takes the last round"

  # ---- the 0.44.0 receipt, whose defects all shipped once ----------------------------------------

  # THE READBACK, CHARGED WHERE IT HAPPENS. `anneal_scheduled` scores the whole state after every
  # sweep to keep a running best; that is a read of the whole state. It was charged by the CALLER,
  # so `Cpu::run` billed it and `Compiled::solve_with` -- the function a caller actually reaches
  # for -- billed nothing, pricing the identical anneal 244x apart. Worse, `KV260_MEASURED` states
  # no `e_read` precisely so a run that read cannot be priced against it, and the model API
  # returned a measured-silicon figure anyway. Set the charge to zero and the receipt test must go
  # red on BOTH the count and the refusal.
  "src/tempering.rs|                l.reads += g.n as u64;|                l.reads += 0;|a_best_of_run_is_charged_for_the_states_it_reads_to_find_the_best|the readback goes uncharged"

  # A BEST-OF SEARCH PAYS FOR EVERY TRY. `best_of_all` sums the ledgers of all N; carrying only the
  # winner's makes an N-restart search read as one run. This shipped across the whole C ABI -- 1, 4
  # and 12 tries all reported 19,200 node updates -- because selection and aggregation lived in two
  # functions and only one of them was fixed.
  "src/model.rs|        winner.cost = total;|        let _ = total;|best_of_n|best-of carries only the winner"

  # THE FABRIC MUST DECLARE THE QUANTISATION IT PERFORMS. `Precision::Fixed` is scale-relative in
  # this crate (step = max|w| / levels); `FixedFabric` uses an absolute Q.8 grid. Under the wrong
  # declaration `check` computed ~0 relative error for a program whose weights were all 0.001 and
  # accepted it, and the fabric quantised every coupling to zero and sampled an empty graph.
  # (Retargeted 2026-09-14: a formatter split the declaration over three lines; the row now swaps
  # the declared variant on its first line and parks the grid literal in a discard.)
  "src/hdl.rs|        f.coupling_precision = Precision::Grid {|        f.coupling_precision = Precision::Fixed { bits: 12 }; let _ = Precision::Grid {|the_declared_step_is_the_one_the_emitter_quantises_on|fabric mis-declares its grid"

  # LOOKING AT THE ALTERNATIVES MUST NOT CHANGE THE RECEIPT. `ft_model_select_optimum` assigns a
  # Solution out of `h.answers`, which holds one per TRY. Without the recompute a 12-try solve
  # reports its full receipt until the caller enumerates optima, at which point agreement silently
  # becomes (1, 1).
  "src/ffi.rs|            s.agreement = s.agreement_among(&h.answers);|            let _ = h.answers.len();|enumerating_optima|selecting an optimum resets the receipt"

  # A RECONFIGURED FABRIC HOLDS NOTHING IT CAN REUSE. The reflash was charged with `if rung > 0` --
  # relative to the CALL -- so a second run's first bitstream was free while the fabric held the
  # previous run's weights and seeds. One write is worth 21,664 node updates at Z1_SPICE, so the
  # missing term was the largest line in the ledger.
  "src/hdl.rs|                self.load_unused = true;|                self.load_unused = false;|a_second_run_pays|second run reflashes for free"

  # A spin in no factor at all. `initial_tables` emits nothing for it and `run` skipped it, so the
  # partition function lost the factor of two that spin contributes. `from_ising` had the identical
  # hole, so the two engines agreed on the wrong number to the last ulp -- which is why this needs a
  # mutation row and not a cross-check: the cross-check was there and was mutually wrong.
  "src/exact.rs|constant -= core::f64::consts::LN_2;|constant -= 0.0;|a_spin_in_no_factor_still_doubles|a free spin's factor of two"

  # ground_degeneracy extrapolates from TWO temperatures, and with both the same there is no
  # extrapolation to do -- (0.0, 0.0) confidently reported 2^n ground states for a model that has
  # two. The guard needs them distinct, and `>=` is the one-character version of not having it.
  "src/exact.rs|&& cold > warm;|&& cold >= warm;|a_temperature_that_says_nothing|equal temperatures certify a non-convergence"

  # A Wolff sweep of "as many steps as it takes to visit n spins" makes the step count a function of
  # the cluster sizes, hence of the state: ordered configurations end the sweep sooner, so sweeps end
  # preferentially just after a large flip. Optional stopping, with every individual move exactly
  # correct. On ring(10) at beta 0.4 it returned <E> = -4.3262 against an enumerated -3.8009.
  "src/cluster.rs|for _ in 0..self.wolff_steps {|while acc.visited < self.gauged.n as u64 {|both_moves_sample_the_distribution|a Wolff sweep whose length reads the state"

  # GraphBuilder sums duplicate pairs and keeps the result, so `couple(0,2,1.0)` then
  # `couple(0,2,-1.0)` leaves a stored edge of weight zero. Without the guard `w > 0.0` is false and
  # the sign rule reads it as antiferromagnetic, manufacturing a frustration the model does not have
  # and refusing a perfectly samplable instance with a "frustrated" cycle of product zero.
  "src/cluster.rs|if w == 0.0 {|if false {|a_zero_coupling_constrains_nothing|a zero coupling read as antiferromagnetic"

  # The interval on `beta` used the inverse Hessian, which is the variance of a REAL likelihood.
  # Pseudolikelihood is a COMPOSITE one and needs the Godambe sandwich. Measured on exact independent
  # draws -- no sampler, so nothing can be wrong but the instrument -- the naive form missed the true
  # beta on 12-18% of runs against the 5% a 95% interval allows, its standard error a stable 0.76 of
  # the right one. `certify` was accusing correct samplers about one run in seven.
  "src/certify.rs|let var = if clusters >= 2 { j / (h * h) } else { 1.0 / h };|let var = 1.0 / h;|the_interval_covers|a pseudolikelihood interval without its sandwich"

  # The autocorrelation inflation, which no test covered until a mutation said so. It cannot be
  # pinned by a coverage test -- it OVER-widens, so removing it moves nominal coverage toward the
  # target rather than away -- so the row is scored against the contract it actually has: a more
  # correlated chain must get a wider interval.
  "src/certify.rs|let half = 1.96 * se * inflate;|let half = 1.96 * se;|a_more_correlated_chain|an interval that ignores autocorrelation"

  # Multi-spin coding packs 64 replicas into one word. Broadcast one coin across the lanes and all 64
  # become the same chain -- and every one of them stays an exactly correct sample of the model, so
  # every per-replica check still passes while the sampler delivers a sixty-fourth of what it claims.
  "src/multispin.rs|let u = self.rng.next_u64();|let u = if self.rng.next_u64() & 1 == 1 { u64::MAX } else { 0 };|the_replicas_are_independent|64 replicas that are one replica"

  # Wang-Landau halved its modification factor on a flat histogram, which is the original recipe and
  # does not converge: the error SATURATES, because whatever statistical error a stage ends with is
  # frozen into ln g when f drops. On a 4x4 lattice the worst error went 0.2392, 0.2147, 0.2110,
  # 0.2104 as the target went 1e-4 to 1e-7 -- three orders of magnitude of work for nothing. The
  # test tolerance sits between the two schedules on purpose, so this row is what enforces the fix.
  "src/wanglandau.rs|if ln_f <= self.graph.n as f64 / steps.max(1) as f64 {|if false {|the_density_of_states_matches_enumeration|a modification factor that only halves"

  # The classic Wang-Landau mistake, and the one the algorithm cannot notice itself: the update
  # belongs to the state the walk is IN, and a rejected proposal leaves it in one.
  "src/wanglandau.rs|let here = self.level_of(self.e)?;|let here = b;|the_derived_quantities_follow_from_the_same_estimate|only accepted moves recorded"

  # Ishikawa's positive-monomial identity takes floor((k-1)/2) auxiliaries. A bare 1 survived every
  # end-to-end reduction test, because the two AGREE at arities three and four and an arity-five
  # factor needs sixteen auxiliaries -- past what an exhaustive minimisation can enumerate. The
  # identity is now checked one monomial at a time, where arity five needs two, and that is the test
  # this row names.
  "src/reduce.rs|        let aux = (k - 1) / 2;|        let aux = 1;|the_penalty_free_identities_hold_at_every_arity|Ishikawa with one auxiliary at every arity"

  # The penalty-free path reports the offset it applied. Reporting it with the wrong sign passes
  # every state-ordering check -- the reduction still moves no state relative to any other -- and
  # lands the caller exactly twice the offset away from the original energy.
  "src/reduce.rs|        penalty: 0.0,|        penalty: 1.0,|the_penalty_free_reduction_moves_no_state_at_any_arity|a penalty-free reduction that reports a penalty"

  # sqa shipped M=4 with beta=10 and gamma_max=3: a Trotter ratio of 7.5, where tanh(7.5) ~ 1 makes
  # the slice coupling vanish and the slices decouple into independent classical replicas. Its own
  # single-spin oracle put the simulated magnetisation at 0.987 against a true quantum 0.316, and at
  # identical work the default found the planted optimum 3 times in 30 where the plateau found 27.
  "src/sqa.rs|            trotter: Params::slices_for(beta, gamma_max),|            trotter: 4,|the_derived_slice_count_beats_the_literal_it_replaced|a Trotter slice count that ignores beta"

  # `adaptive` shipped eight replicas over a span of eighty in beta. On a 14x14 planted glass that
  # leaves three adjacent pairs at or below 0.01 -- a ladder cut in half by the module's own stated
  # criterion -- and `adapt` does not repair it, because respacing moves interior rungs while the
  # endpoints are held. The count needed grows as sqrt(n), which is the rule this row protects.
  "src/adaptive.rs|    let r = (0.35 * (n as f64).sqrt() * span).ceil().min(4096.0) as usize;|    let r = (0.35 * span).ceil().min(4096.0) as usize;|the_derived_replica_count_gives_a_ladder_that_is_not_severed|a replica count blind to the model size"

  # A certified LOWER bound that rounds up is not a bound. `verify` summed the dual point with
  # `iter().sum()`, and left-to-right addition can round upward: [1.0, 3*2^-54, 3*2^-54] sums over
  # its true value by 1.1e-16. The points `certified` produces are on a power-of-two grid where the
  # sum is exact, but `verify` exists for certificates it did NOT produce, and `y` is a public field.
  "src/sdp.rs|        Ok(sum_down(&self.y))|        Ok(self.y.iter().sum())|verify_sums_the_dual_point_downward|a certified bound summed in the wrong direction"

  # The defect this module was written for. `forest` summed its parts with `+=`, and on random
  # trees -- where the bound is EXACT and the gap must be zero -- 1688 of 4800 trials reported a
  # negative gap. Every one of them passed the existing `within 1e-9 of the optimum` check, which
  # is five orders of magnitude too loose to see it.
  "src/bound.rs|        let total = sum_down(&part_energies) - sum_up(&guards);|        let total = part_energies.iter().sum::<f64>();|no_bound_is_ever_above_a_state_it_bounds|a lower bound accumulated with plain addition"

  # Soundness here is a property of the PAIR: the bound must round down AND the energy up. Fixing
  # only the bound is worse than fixing neither -- `decoupled` accumulates in exactly
  # `Graph::energy`'s order, so its two roundings had been cancelling, and 0 negative gaps became
  # 78 the moment the bound alone became sound.
  "src/bound.rs|        sum_up(&[energy_up(g, s), -self.value])|        g.energy(s) - self.value|no_bound_is_ever_above_a_state_it_bounds|a gap measured from an energy that rounds either way"

  # The direction itself. `total + guard` is still a compensated sum, still accurate, and points
  # the wrong way -- which no accuracy check can distinguish from the right way.
  "src/round.rs|    total - guard|    total + guard|a_sum_that_plain_addition_rounds_upward_is_bracketed|a downward sum that rounds upward"

  # A clause is violated when every literal is false, and the subset expansion of that indicator has
  # an empty-subset constant. Dropping it leaves every optimum right and every reported number off by
  # a fixed amount -- an error that passes a solver test and only fails against a published result.
  "src/dimacs.rs|                    offset += term;|                    offset += 0.0 * term;|the_energy_is_the_unsatisfied_weight_at_every_assignment|a MAX-SAT translation that drops its constant"

  # Pseudolikelihood's gradient: an edge weight enters the field of BOTH its endpoints, so its
  # derivative has two terms. Drop either and the fit still converges to something, just not to the
  # maximiser of the stated objective -- which no check on the fitted model would reveal.
  "src/ebm.rs|            dw[e] += resid[i] * f64::from(row[j]) + resid[j] * f64::from(row[i]);|            dw[e] += resid[i] * f64::from(row[j]);|the_closed_form_gradient_matches_a_finite_difference|a pseudolikelihood gradient missing half its edge term"

  # The three sampler-free losses share one gradient and differ only in a slope. `exp(-u)` with the
  # sign dropped is still a smooth positive function of the margin -- it is just INCREASING in it,
  # so the fit climbs away from the data while every objective curve rises.
  "src/ebm.rs|            FlipLoss::Flow => (-u).exp(),|            FlipLoss::Flow => u.exp(),|every_flip_loss_pushes_the_margin_up|a flip loss that rewards a small margin"

  # Ratio matching's slope is 4q^2(1-q), the derivative of -q^2 through q = sigma(-2u). Written as
  # 4q(1-q) it is the derivative of something else and still positive, still smooth, still
  # convergent -- a gradient ascent on a function nobody stated.
  "src/ebm.rs|                4.0 * q * q * (1.0 - q)|                4.0 * q * (1.0 - q)|every_flip_gradient_matches_a_finite_difference|a ratio-matching slope off by a factor of q"

  # The exact ML gradient's negative phase is a signed sum: a state contributes +p when the two
  # endpoints agree and -p when they differ. Dropping the sign leaves the partition function
  # correct and every correlation equal to one, which is a gradient that still points somewhere.
  "src/ebm.rs|            model_w[k] += if (mask >> i & 1) == (mask >> j & 1) { p } else { -p };|            model_w[k] += p;|the_exact_gradient_matches_a_finite_difference_of_the_true_likelihood|an exact negative phase that ignores the spin product"

  # The positive phase averages over the states that AGREE WITH A ROW ON ITS VISIBLE PART, which is
  # the sum a clamped sampler estimates. Dropping the latent completion leaves the fully-visible
  # case exactly right -- it has one completion -- and silently wrong on every model with a hidden
  # unit, which is every model anyone fits.
  "src/ebm.rs|    let comps = 1usize << (n - data.visible);|    let comps = 1;|the_exact_gradient_matches_a_finite_difference_of_the_true_likelihood|an exact positive phase that never visits the latent states"

  # The variational positive phase is the MEANS of a clamped mean field. Using the sampled state
  # instead turns `train_variational` back into `train` with extra steps -- it still fits, still
  # raises the likelihood, and is no longer the method it is named for.
  "src/ebm.rs|                    d_edge[e] += mf.m[i] * mf.m[j]|                    d_edge[e] += 0.0 * mf.m[i] * mf.m[j]|a_fully_visible_variational_fit_has_an_exact_positive_phase|a variational positive phase that carries no data"

  # The clamp must PIN. Letting a clamped spin iterate makes the positive phase an average over
  # states that disagree with the data -- a different objective under the same name, and one that
  # still converges to something.
  "src/meanfield.rs|        for i in fixed..g.n {|        for i in 0..g.n {|the_clamped_mean_field_pins_what_it_is_told|a clamp that does not clamp"

  # The barrier is what makes the parallel sweep SAFE, not just fast: within a colour class no two
  # nodes are adjacent, but class c+1 reads what class c wrote. A barrier that lets one thread run
  # ahead still produces plausible spins, and only a check on the VALUES catches it.
  "src/barrier.rs|            if spun < self.spin {|            if spun < self.spin { return; } else if false {|nothing_crosses_a_crossing|a barrier that returns before everyone arrives"

  # The stream derivation is what the bit-identity guarantee rests on. Shifting the class index by
  # one bit still gives every (sweep, class, chunk) a distinct stream -- the sampler still samples,
  # the physics test still passes, and the numbers are silently different from every prior run.
  "src/gibbs.rs|                                        ^ (ci as u64) << 32,|                                        ^ (ci as u64) << 33,|parallel_sweeps_are_bit_identical_to_the_old_spawn_per_sweep_shape|a parallel RNG stream derived differently"

  # `g(x) = x g(1/x)` is the whole reason the energy cancels out of the acceptance. Break it and the
  # chain still runs, still converges, and converges to the wrong distribution -- which only a check
  # against enumeration can see.
  "src/informed.rs|            Balance::Sqrt => 0.5 * log_r,|            Balance::Sqrt => log_r,|every_balancing_function_is_balanced|a balancing function that is not balanced"

  # Accepting everything is a valid-looking sampler with no Metropolis correction at all. It mixes
  # FASTER, which is the trap: a speed measurement would reward it.
  "src/informed.rs|        let alpha = (before / after).min(1.0);|        let alpha = 1.0;|the_informed_chain_samples_the_boltzmann_distribution|an informed chain that accepts everything"

  # The proposal is normalised by a total maintained incrementally. Flip the sign of the field
  # repair and the weights describe a state the sampler is not in.
  "src/informed.rs|            self.fields[j] += self.g.w[e] * 2.0 * sk;|            self.fields[j] -= self.g.w[e] * 2.0 * sk;|the_incremental_total_agrees_with_a_recomputation|an incremental field update with the wrong sign"

  # The shift centres the weights. Bounded over the whole MODEL instead of centred on the weights
  # actually present, a single pinned variable -- which is what clamping is -- underflows every
  # other site to zero and the chain stops dead. It shipped that way; `factor_by_sampling` scored
  # 0 of 45 against plain Gibbs's 36 before this line changed.
  "src/informed.rs|        self.shift = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);|        self.shift = logs.iter().copied().fold(f64::INFINITY, f64::min);|a_pinned_variable_does_not_stall_the_rest_of_the_model|a shift centred on the smallest weight instead of the largest"

  # The gate penalty is a SIGNED expansion of the indicator over subsets. Drop the signs and it is
  # still a polynomial, still has a ground state, and encodes a different relation entirely.
  "src/invertible.rs|                            if a >> i & 1 == 0 {|                            if false {|the_penalty_counts_violated_gates_exactly|an indicator expansion with the signs dropped"

  # An array multiplier's final carry is one line and one bit. Dropping it gives a circuit that is
  # a perfectly good Ising model computing the wrong product for exactly the operand pairs that
  # overflow -- which is why the test enumerates every pair rather than sampling some.
  "src/invertible.rs|            next.push(ci);|            let _ = ci;|a_multiplier_computes_products_exhaustively|a multiplier that drops its final carry"

  # Backward sampling is backward for a reason: a variable's conditional scope holds only variables
  # eliminated AFTER it, so walking the order forwards conditions on spins that have not been drawn.
  # The chain still produces states, still with a plausible energy histogram, from another
  # distribution entirely.
  "src/exact.rs|        for step in self.steps.iter().rev() {|        for step in self.steps.iter() {|backward_sampling_draws_the_boltzmann_distribution|a backward pass walked forwards"

  # The tables hold ENERGIES, so ln P(+1) - ln P(-1) is branch[0] - branch[1]. Flipped, every spin
  # is drawn from its own mirror image -- a perfectly normalised distribution, and the wrong one.
  "src/exact.rs|                logit[idx] = branch[0] - branch[1];|                logit[idx] = branch[1] - branch[0];|backward_sampling_draws_the_boltzmann_distribution|conditional log-odds with the sign flipped"

  # A spin in no factor is uniform and still has to be DRAWN. `log_partition` only owes it ln 2, so
  # the obvious reading of its `continue` is that a sampler owes it nothing -- and then it reports
  # whatever the state vector was initialised to.
  "src/exact.rs|                steps.push(Step { v, scope: Vec::new(), logit: vec![0.0] });|                let _ = v;|a_spin_in_no_factor_is_still_drawn|a free spin never drawn"

  # The same line as seen by the saddle-point claim: with the hidden bits read off a visible-only
  # key they are pinned at -1 rather than averaged to zero, so an RBM at zero weights acquires a
  # gradient it does not have and the stationary point disappears.
  "src/ebm.rs|                row_w[k] += if (mask >> i & 1) == (mask >> j & 1) { p } else { -p };|                row_w[k] += if (vkey >> i & 1) == (vkey >> j & 1) { p } else { -p };|an_rbm_at_zero_weights_is_a_stationary_point_of_the_exact_likelihood|hidden units pinned instead of averaged in the positive phase"

  # Onsager's energy density has its elliptic pole and its vanishing coefficient at the SAME point,
  # sinh(2K) = 1. At beta_c plus one ulp sinh rounds to exactly 1.0, so the naive expression is
  # 0 * infinity and the exact oracle returns NaN. At beta_c itself it works by luck.
  "src/free_energy.rs|    let term = if comp == 0.0 { 0.0 } else { coeff * elliptic_k_comp(comp) };|    let term = coeff * elliptic_k_comp(comp);|the_exact_pole_is_reachable_and_is_handled|an exact oracle that is NaN at criticality"

  # The vector LQR oracle. A Riccati iteration converges to SOMETHING however it is derived, so the
  # load-bearing check is putting the answer back into the equation -- and the load-bearing claim is
  # that x0^T P x0 is what the optimal policy actually spends, not a bound on it.
  "src/mppi.rs|                next[i] = s.q[i] + atpa[i] - corr[i];|                next[i] = s.q[i] + atpa[i];|the_matrix_riccati_solution_solves_the_equation|a Riccati solution that is not one"

  # Slicing sums over pinned assignments, and `pin` was written for MARGINALS where its dropped
  # constants cancel in a ratio. Summing them does not cancel: the pinned node's own field must come
  # back as +beta*h*v, read from the graph at the time of the pin because earlier pins fold into it.
  # Zero on a field-free model, which is why the fixtures carry a field on every node.
  "src/exact.rs|                Some(b) => b * src.h[i] * v - core::f64::consts::LN_2,|                Some(b) => b * src.h[i] * v,|slicing_agrees_with_the_direct_computation|a sliced partition function missing its free spin"

  # A portfolio's arms have to be asked the SAME question. Handing each the full budget rather than
  # a share makes the portfolio look free -- it beats any single arm at "the same" cost, having spent
  # k times as much -- and no comparison between the arms reveals it.
  "src/portfolio.rs|    let each = budget.split(arms.len());|    let each = budget;|the_budget_is_divided_among_the_arms|a portfolio that gives every arm the whole budget"

  # A conflict set has to contradict itself in isolation, or it is a list rather than a witness. The
  # ExactlyOne closures pushed an EMPTY provenance, so a fix following from other literals being
  # false forgot what made them false -- and the reported set no longer stood on its own.
  "src/model.rs|                                    push(v, x, why.clone());|                                    push(v, x, Vec::new());|the_conflict_set_contradicts_itself_in_isolation|a conflict set that does not prove itself"

  # Propagating a SOFT constraint fixes a variable a solution is allowed to disagree with, which is
  # the same unsoundness as inventing a constraint -- and invisible in any model whose soft rows
  # happen to hold at the optimum.
  "src/model.rs|                if !*hard {|                if false {|a_soft_constraint_forces_nothing|a presolve that propagates soft constraints"

  # A receipt exists to be re-checked. If `verify` reads the stored energy instead of recomputing it
  # from the state, every test still passes and nothing is verified -- the check becomes a
  # restatement of the claim it was meant to test.
  "src/receipt.rs|        let actual = g.energy(&self.state);|        let actual = self.energy;|no_single_field_can_be_edited_and_still_pass|a receipt that restates rather than checks"

  # ---------------------------------------------------------------------------------------------
  # Twenty-four rows for the twenty-four modules added in the two algorithm waves. Each was
  # applied, SEEN RED against the test it names, and restored before it was written down. They
  # exist because 300 tests arrived at once, and this crate has already learned that a test which
  # has never been shown to fail is a claim, not a measurement.
  # ---------------------------------------------------------------------------------------------

  # The one place the algorithm asks whether the two sandwiching chains actually MET. A
  # one-character index typo makes the question always true, so `draw` breaks out of its doubling
  # loop on the first attempt and the perfect sampler degrades into an ordinary forward chain from
  # a fixed start with a burn-in of one sweep -- silently, for every seed.
  "src/cftp.rs|(chains[0] == chains[1]).then(|(chains[0] == chains[0]).then(|every_start_lands_on_the_coalesced_state|a coalescence check that always holds"

  # Gillespie's waiting time is the only place the chain stops being an estimator and becomes a
  # CLOCK. Advancing by 1/R instead of -ln(u)/R is the dwell's own mean, so occupancy stays
  # unbiased and eight of nine kmc tests still pass -- including the Boltzmann oracle. The row is
  # therefore also evidence about WHICH line of the named test bites: its mean assertion passes
  # exactly under the mutation, and only the CDF check sees it.
  "src/kmc.rs|        let dwell = -u.ln() / total;|        let _ = u; let dwell = 1.0 / total;|the_waiting_time_is_exponential_in_the_total_rate|a deterministic clock instead of an Exp(R) draw"

  # Hazan & Jaakkola's all-dimensions bound splits gamma_i(s_i) into a_i + b_i s_i so the draw is
  # one MAP solve with shifted fields. Dropping the constant a_i is the textbook error: the fields
  # still shift, the maximiser is still found, and the bound is no longer one.
  "src/perturb.rs|            consts[i] = 0.5 * (gp + gm);|            consts[i] = 0.0;|the_all_dimensions_bound_is_exactly_log_z_when_nothing_is_coupled|an all-dimensions perturbation without its constant"

  # The edge appearance probability rho_e is what makes TRW an UPPER bound. Drop it and the entropy
  # subtracts the full mutual information -- that is the Bethe free energy, which sits on EITHER
  # side of ln Z. The module's header warns about exactly this accident.
  "src/trw.rs|entropy -= cover.rho[e] * (entropy_of(&bi) + entropy_of(&bj) - entropy_of(&b));|entropy -= entropy_of(&bi) + entropy_of(&bj) - entropy_of(&b);|bounds_exact_log_z_from_above|TRW entropy without the edge appearance weight"

  # This line is the whole of g(E) T(E->E') = g(E') T(E'->E) in logs, and each raw count must be
  # divided by ITS OWN source level's attempt total. Transposing the two totals is an a/b typo in a
  # symmetric-looking expression that still returns a full, finite density of states.
  "src/tmmc.rs|                let d = (cab / visits[a]).ln() - (cba / visits[b]).ln();|                let d = (cab / visits[b]).ln() - (cba / visits[a]).ln();|the_solver_is_exact_on_the_enumerated_matrix|row totals taken from the wrong level in the detailed-balance solve"

  # Grassberger 1988's G(n) = psi(n) + (-1)^n / (n+1). Writing (-1)^(n+1) is the transcription
  # error people actually make: the correction keeps its SIZE and points the wrong way. Of 1212 lib
  # tests, one notices.
  "src/entropy.rs|        let sign = if c % 2 == 0 { 1.0 } else { -1.0 };|        let sign = if c % 2 == 0 { -1.0 } else { 1.0 };|the_grassberger_correction_moves_toward_its_own_exact_target|Grassberger oscillating term sign"

  # The noise-to-data ratio enters NCE twice. One appearance is inside the log-odds and is easy to
  # test; the other is the MIXTURE WEIGHT, because the population objective averages the noise term
  # over nu*D draws. Dropping it is what you get by writing the objective from the picture instead
  # of the expectation.
  "src/nce.rs|        acc += data_probability[mask] * log_sigmoid(delta) - ratio * ln_pn.exp() * softplus(delta);|        acc += data_probability[mask] * log_sigmoid(delta) - ln_pn.exp() * softplus(delta);|the_truth_maximises_the_population_objective|a population objective whose noise class is not weighted by the ratio"

  # `e ^ 1` is the paired-arc trick that turns an out-arc into the arc pointing back in, and
  # `reach_backward` is the sink-side half of the strong-persistency reading. Without it the walk
  # still terminates and still returns a set -- one that no longer means "every minimiser agrees".
  "src/roofdual.rs|                let back = e ^ 1;|                let back = e;|the_pinned_set_is_what_every_minimiser_of_the_relaxation_agrees_on|persistency reads the wrong arc"

  # Wishart planting projects out the planted direction, and t has squared norm n, not 1. Forgetting
  # the 1/n leaves a projection that still produces a symmetric instance with a plausible spectrum
  # and a ground energy that no longer matches the closed form it was built to have.
  "src/planted.rs|            *x -= dot * t[i] as f64 / n as f64;|            *x -= dot * t[i] as f64;|the_closed_form_ground_energy_is_right|Wishart planting: the projection forgets that t has squared norm n"

  # The negative log-likelihood of a word is -sum_i L_i s_i / 2, so h_i = L_i / 2. This is the only
  # place that half appears, and "the field is the LLR" reads correctly to anyone who has not done
  # the derivation.
  "src/ldpc.rs|            bias[i] += l / 2.0;|            bias[i] += l;|the_energy_is_the_penalty_plus_the_channel_at_every_state|the channel field loses its half"

  # `est ^ xm` scores the estimate against the TRUE image; `est ^ ym` scores it against the NOISY
  # one. Two live loop variables one character apart, and with the wrong one the Nishimori claim is
  # decided against the observation -- which is the thing the condition is about.
  "src/restore.rs|                let wrong = (est ^ xm).count_ones();|                let wrong = (est ^ ym).count_ones();|nishimori_minimises_the_exact_expected_error|the joint scores the restoration against the observation, not the truth"

  # The module exists for one formula: torque grows like sqrt(<J^2> d). Dropping the root leaves a
  # rule LINEAR in degree -- the exact alternative its first paragraph argues against -- which
  # compiles, returns a positive strength, and is indistinguishable from correct on a 2-regular
  # model.
  "src/torque.rs|    prefactor * rms_coupling(logical) * mean_degree(logical).sqrt()|    prefactor * rms_coupling(logical) * mean_degree(logical)|on_spiked_cliques_the_rms_rule_beats_the_max_rule|chain strength linear in degree, not its root"

  # N_k in the mixture denominator is what makes MBAR a JOINT solve rather than chained pairs: a
  # sample is credited in proportion to how many draws each state actually contributed, and the
  # exactness proof turns on the same N_k appearing there. Uniform weights leave a plausible number.
  "src/mbar.rs|            acc += (ln_n[j] + f[j] - u[j][n] - mx).exp();|            acc += ((out.len() as f64 / k as f64).ln() + f[j] - u[j][n] - mx).exp();|exact_empirical_distributions_are_solved_exactly|an MBAR mixture weighted uniformly instead of by sample count"

  # The module's own documented ergodicity trap, put back. With a FIXED length every proposal flips
  # a number of spins of one parity, so at even lengths up-spin parity is conserved and half the
  # state space is unreachable. The kernel stays exactly reversible -- invariance, the MH-definition
  # check and the bookkeeping check all stay green. INVARIANCE DOES NOT IMPLY IRREDUCIBILITY.
  "src/multiflip.rs|        let len = 1 + (self.rng.next_u32() as usize) % self.length;|        let len = self.length;|the_path_chain_samples_the_boltzmann_distribution|a fixed path length instead of a uniform draw from 1..=length"

  # The discrete Stein operator is applied in BOTH arguments; `sy[i] * (k - k_x)` is the y-side
  # application, and one character makes it the operator applied twice in x. The kernel stays
  # finite, stays plausible, and stops being a Stein kernel.
  "src/stein.rs|        acc += sx[i] * sy[i] * k - sx[i] * (k - k_y) - sy[i] * (k - k_x)|        acc += sx[i] * sy[i] * k - sx[i] * (k - k_y) - sx[i] * (k - k_x)|the_stein_kernel_is_symmetric|the Stein operator applied twice in x and never in y"

  # Every intra-part coupling is already inside its part's exact ln Z_p -- that IS the module's
  # claim, the tractable subgraph solved exactly. Counting it again in the mean-field term charges
  # it twice, and the result is not a looser bound, it is not a bound.
  "src/structured.rs|                if j > i && of[j] != of[i] {|                if j > i && of[j] == of[i] {|the_bound_is_below_log_z_at_arbitrary_fields|intra couplings counted twice, so the bound sits above ln Z"

  # This coefficient IS the moral edge: the co-parent coupling a conversion that forgot to moralise
  # would never emit. Flipping its sign is the E = -w s s versus +w s s slip, and it leaves a graph
  # with the right STRUCTURE and the wrong explaining-away.
  "src/bayes.rs|                        0.25 * (g[0] - g[1] - g[2] + g[3]),|                        -0.25 * (g[0] - g[1] - g[2] + g[3]),|deleting_the_moral_edge_changes_the_marginal|a moral edge that couples the wrong way"

  # RESYNC_SLACK gates when to spend an exact recompute; SOLVED_TOL decides whether the recomputed
  # answer is a hit. Reusing the gate constant one line down is the most natural wrong version of
  # this function -- same comparison, same variable, a thousand times looser.
  "src/rld.rs|                    if e <= self.target + SOLVED_TOL {|                    if e <= self.target + RESYNC_SLACK {|every_reported_hit_is_a_state_that_is_actually_at_the_optimum|a hit answered from the resync slack"

  # The SPIN->BINARY substitution contributes offset -= sum_i h_i. The INVERSE direction, sixteen
  # lines below in the same match, legitimately writes `offset += *h`, so the mutant is a
  # copy-paste between branches -- and an offset error ranks every state identically.
  "src/dimod.rs|                    offset -= *h;|                    offset += *h;|the_offset_is_invisible_to_the_ranking_and_fatal_to_the_value|vartype constant with the wrong sign"

  # `coupler_live` is deliberately NOT read off the working graph -- it is the one place the defect
  # map is consulted directly, and the logical-edge check, chain connectivity and "what would this
  # run program" all rest on it. Consulting only the datasheet returns the machine in the brochure.
  "src/working.rs|        self.ideal_couplers.contains(&key(a, b)) && !self.defects.coupler_dead(a, b)|        self.ideal_couplers.contains(&key(a, b))|the_audit_names_the_defect_that_breaks_a_placement|coupler_live consults the datasheet only"

  # `pick` is the POSITION inside the shrinking candidate list; `i` is the spin. `flips` is the only
  # record of which spins were flipped and in what order, and it is what makes "the best state
  # between these two" a claim anyone can replay.
  "src/relink.rs|        flips.push(i);|        flips.push(pick);|a_path_is_its_start_plus_exactly_the_flips_taken|a flip order recorded by position instead of spin"

  # The replica exchange must be accepted with exp((beta_j - beta_i)(E_j - E_i)). Flipping the
  # energy difference still produces swaps and still reports plausible swap rates, and the cold
  # chain is then stationary for nothing -- which is the only thing the negative phase reads.
  "src/tempered_cd.rs|            let arg = (self.betas[i + 1] - self.betas[i]) * (e_j - e_i);|            let arg = (self.betas[i + 1] - self.betas[i]) * (e_i - e_j);|the_cold_end_of_a_swapped_ladder_is_a_sample_from_the_model|replica swap criterion sign"

  # The Fisher information IS Var(E), so the line element is sqrt(Var(E)). Dropping the root gives
  # the thermodynamic DIVERGENCE, which Crooks 2007 defines on the same page as the length, under a
  # different name -- a positive, finite, integrable, wrong ladder.
  "src/length.rs|    variance(dos, beta).sqrt()|    variance(dos, beta)|the_equal_length_ladder_is_the_inverse_gudermannian_on_free_spins|thermodynamic length without its square root"

  # The clamp is why `runs_needed` is a function and not an expression. For p >= s the closed form
  # returns LESS than one run, so a solver that already clears the target in a single run is billed
  # 0.67 t. That is the direction that flatters, and it is worst where the solver is doing best.
  "src/tts.rs|        return Some(1.0);|        return Some((1.0 - s).ln() / (1.0 - p).ln());|runs_needed_matches_the_closed_form_and_clamps_where_it_must|TTS reports a fraction of a run"

  # ---------------------------------------------------------------------------------------------
  # Thirteen rows for the thirteen algorithms of wave 3, each produced by the agent that wrote the
  # module, applied, SEEN RED, and restored before it was recorded. Three of them are evidence
  # about the TESTS rather than the code, and those three are the reason the row is written at the
  # same time as the module rather than a wave later: `sse`, where the first mutation survived
  # because the acceptance rule it broke was unreachable from any test; `vmc`, where the assertion
  # the physics guarantees stayed silent and a pointwise identity caught it; and `survey`, where
  # both solver-level tests pass with a wrong survey equation because the fallback rescues them.
  # ---------------------------------------------------------------------------------------------

  # The coupling term is the only thing that tells the oscillator network which way DOWN is. With
  # its sign flipped the machine is a perfectly good Ising MAXIMISER: it still settles, still
  # binarises, still reports a state and an energy. The row also records what the module doc now
  # says out loud -- the single-instance asymmetric test does NOT separate the real CAC law from
  # the plausible simplification that drops the multiplicative e_i. Only the twenty-instance family
  # test does, and only because both counts are frozen: "corrected beats plain" would have passed.
  "src/cim.rs|let dx = (pump - 1.0 - xi_now * xi_now) * xi_now + self.xi * e[i] * drive[i];|let dx = (pump - 1.0 - xi_now * xi_now) * xi_now - self.xi * e[i] * drive[i];|two_spin_ferromagnet_and_frustrated_triangle_match_exhaustive_enumeration|coupling term sign flipped, so the machine maximises the Ising energy instead of minimising it"

  # sin(2 phi) is what makes this an ISING machine; sin(phi) injection-locks at the fundamental and
  # leaves an XY machine that never binarises. The closed form tan(phi(t)) = tan(phi_0) exp(-2 K_s t)
  # is what sees it, and the same test checks that halving dt halves the error, so a wrong factor
  # inside the sine cannot converge prettily to the wrong number.
  "src/oim.rs|            let d = -(self.k * c + k_s * (2.0 * phi[i]).sin());|            let d = -(self.k * c + k_s * phi[i].sin());|shil_relaxation_matches_the_closed_form_tan_decay|SHIL term loses its sub-harmonic factor of two, injection-locking at the fundamental instead"

  # Pi(u) and Pi(s) swapped: a variable counted as forced TOWARD satisfying the clause instead of
  # away from it. Only 2 of the module's 11 tests go red under it, and the row exists to record
  # which 2: BOTH decimation tests pass with a wrong survey equation, because the local-search
  # fallback rescues the solver and the surveys stay non-trivial either way. The solver-level tests
  # are not what keeps the physics honest here; the tree enumeration and the closed form are.
  "src/survey.rs|        let forced_away = (1.0 - disagree) * agree;|        let forced_away = (1.0 - agree) * disagree;|survey_propagation_is_the_backbone_by_exhaustive_enumeration_on_trees|Pi(u) and Pi(s) swapped: a variable counted as forced TOWARD satisfying the clause instead of away from it"

  # The Moebius recursion c_R = 1 - sum over ANCESTORS is what makes each variable counted exactly
  # once. Adding instead of subtracting still yields integer counting numbers and a region graph
  # that runs -- and an approximation that is no longer a free energy.
  "src/region.rs|                c -= counting[a];|                c += counting[a];|region::tests::counting_numbers_count_every_variable_and_coupling_exactly_once|Moebius recursion adds ancestor counting numbers instead of subtracting them"

  # The cavity is the site's own contribution DIVIDED OUT, which in natural parameters is a
  # subtraction. Adding instead still gives a well-formed Gaussian and still converges; it is just
  # tilted by two copies of the site instead of none.
  "src/ep.rs|    (1.0 / sigma - lam, mean / sigma - gam)|    (1.0 / sigma - lam, mean / sigma + gam)|ep_is_exact_on_an_uncoupled_model_oracle_tanh_closed_form|cavity field added instead of divided out (sign of the natural-parameter subtraction)"

  # The Lagrangian charges a violated triangle AGAINST the bound. Adding the penalty instead leaves
  # a number that rises with every cut and reads like a tightening bound while ceasing to be one.
  # Note what the same agent measured and turned into a second test: stopping as soon as separation
  # finds nothing -- the textbook rule -- made the bound WORSE the more cuts it was offered
  # (C_11 gap 6.6e-3 with 64 rows, 1.6e-1 with 256), because separation reads an approximately
  # optimal primal average and goes quiet while the dual is still climbing.
  "src/cuts.rs|self.outer.push(-rhs * l);|self.outer.push(rhs * l);|k4_is_loose_in_the_box_and_exact_after_triangles_against_enumeration|Lagrangian penalty added instead of subtracted: sign slip on lambda_t * b_t"

  # The sum-of-squares residual is what the certificate is CHARGED, and adding it instead turns a
  # lower bound into a number above the optimum. The enumeration test is the one that sees it,
  # which is the same shape as the unsound `bound::forest` this crate shipped once already.
  "src/sos.rs|parts.push(-miss);|parts.push(miss);|sos::tests::the_bound_never_exceeds_the_true_minimum_by_enumeration|the sum-of-squares residual added to the bound instead of charged against it"

  # `better_than` decides which live point is the WORST and therefore gets deleted. Reversed, the
  # run peels UP the spectrum instead of down: the prior volume still shrinks as exp(-i/N), the
  # shrinkage statistics still check out, and ln Z is computed over the wrong end of the ladder.
  "src/nested.rs|        return a.0 < b.0;|        return a.0 > b.0;|nested::tests::log_z_matches_exhaustive_enumeration_within_its_own_stated_uncertainty|better_than picks the lowest-energy point as worst, so the run peels UP the spectrum instead of down"

  # WHAM's denominator carries +beta f_w -- the free-energy offset that stitches the windows
  # together. With the sign reversed every window's offset is applied backwards, and the stitched
  # profile is still smooth, still normalised, and not the free energy of anything.
  "src/umbrella.rs|            term[w] = ln_n[w] + beta * f[w] - beta * bias[w][b];|            term[w] = ln_n[w] - beta * f[w] - beta * bias[w][b];|wham_recovers_the_enumerated_profile_from_exact_histograms|WHAM denominator uses minus beta f_w instead of plus, so every window offset is applied backwards"

  # THE FIRST ATTEMPT AT THIS MUTATION SURVIVED, and that is why the row is here. The removal
  # acceptance min(1, (L-n+1)/(beta W_max)) clamped to 1 at every point the suite sampled, so the
  # expression was live in the source and DEAD IN THE TESTS -- a whole acceptance rule no test
  # could reach. The binding regime is cold and nearly classical (n > 14, diagonal-dominated), so
  # (beta, Gamma) = (3.0, 0.2) was added to the four-site ring sweep. With that one point present
  # the mutation dies at 13.1 sigma.
  "src/sse.rs|let acc = (l - self.n + 1) as f64 / (self.beta * self.w_max);|let acc = (l - self.n + 1) as f64 / self.w_max;|sse::tests::energy_and_magnetisations_match_exact_diagonalisation_of_a_four_site_ring|beta dropped from the operator-removal acceptance in the diagonal update"

  # AND THE BOUND TEST DID NOT CATCH IT. The variational energy is guaranteed to be an upper bound
  # on E_0, so that assertion looks like the one with teeth -- but flipping the transverse sign
  # makes the reported energy too HIGH, which stays a valid-looking upper bound at every step. What
  # caught it, at 15.09 against 1e-13, was the zero-variance identity: a POINTWISE check with no
  # averaging. The module doc had claimed a sign error "comes out too low and reads as success";
  # that sentence was measured and rewritten.
  "src/vmc.rs|e -= self.gamma * psi.log_ratio_with(s, theta, i).exp();|e += self.gamma * psi.log_ratio_with(s, theta, i).exp();|vmc::tests::the_local_energy_is_flat_at_the_closed_form_product_ground_state|sign of the transverse-field term in the local energy"

  # A gauge multiplies each coupling by g_i g_j. Using one end only leaves a transform that still
  # runs, still inverts on the fields, and no longer preserves the energy spectrum. The named test
  # enumerates all 2^14 states and asserts EXACT f64 equality state by state, because the whole
  # point of a gauge is that it changes nothing.
  "src/gauge.rs|let sign = f64::from(self.signs[i]) * f64::from(self.signs[j]);|let sign = f64::from(self.signs[i]);|the_spectrum_is_invariant_under_a_gauge_over_every_state_exactly|gauge the coupling from one end only, dropping g_j"

  # `q_step(t, j, xt)` is the probability of landing on xt from j; transposing it reads the D3PM
  # posterior off the wrong direction of the chain. It still normalises, still integrates to a
  # distribution, and only brute-force Bayes from the dense kernel products can tell.
  "src/diffuse.rs|            let num = self.q_step(t, j, xt) * self.q_bar(t - 1, x0, j);|            let num = self.q_step(t, xt, j) * self.q_bar(t - 1, x0, j);|posterior_matches_brute_force_bayes_from_dense_kernel_products|D3PM posterior reads the step matrix transposed"

  # ---------------------------------------------------------------------------------------------
  # Thirteen rows for wave 4, each produced by the agent that wrote the module and SEEN RED before
  # it was recorded. Read the qaoa and hmc rows for what they say about tests: a check that is
  # provably blind to the defect it was written for, and a fixture that is a fixed point of the
  # dynamics under test.
  # ---------------------------------------------------------------------------------------------

  # The objective SENSE. Three of these four corpora write character-for-character the same body
  # line and mean three different things by it; dropping the sign flip on a maximise instance
  # yields a Graph whose ground state is the worst cut. Round-trip and cross-format enumeration see it.
  "src/corpora.rs|gb.couple(i, j, -w);|gb.couple(i, j, w);|corpora::tests::oracle_the_crossing_edge_sum_matches_every_max_cut_instance|drop the max-cut coupling negation in MaxCut::instance, so minimising the energy minimises the cut"

  # The repair must be EXACT on the freed block; a greedy repair still improves, still terminates,
  # and is not LNS. Measured alongside: Shaw's related removal is most of the method, not a
  # refinement -- 34/36 planted instances against 9/36 for random removal at the same budget.
  "src/lns.rs|    -f64::from(v) * field|    f64::from(v) * field|lns::tests::grasp_construction_at_alpha_zero_is_exact_on_a_decoupled_model|GRASP's greedy function loses the crate's E = -J s s - h s sign: the construction picks the WORST value for every variable"

  # The long-term memory factor. Pinning it shut still solves 3 of 6 instances at n = 50 and 1 of
  # 10 at n = 100 where the full flow solves 10/10 -- so a small-instance suite would pass an
  # implementation that dropped it outright. The row's test is written at the size where it shows.
  "src/memcomp.rs|let g = 0.5 * q * other;|let g = 0.5 * other;|memcomp::tests::a_sign_consistent_flow_lowers_every_residual_at_every_step_exactly|drop the literal polarity q from the gradient term G_mn (the sign a careful transcription loses)"

  # The straight-through estimator's backward pass. Recorded alongside: the same agent measured that
  # `program::reinforce_grad` returns the gradient SHRUNK by (1 - 1/episodes), because its
  # batch-mean baseline correlates with each episode's own score -- 21.4 standard errors from
  # exact at B = 8. Not fixed here; see relax::tests::program_reinforce_is_the_exact_gradient_shrunk.
  "src/relax.rs|                - eta * df_cond[i] * drelaxed(zt[i], lambda) * dzt[i];|                - eta * df_cond[i] * drelaxed(zt[i], lambda);|relax::tests::rebar_converges_to_the_closed_form_gradient_and_gumbel_softmax_does_not|REBAR drops dz_tilde/dtheta: the conditional relaxation's own dependence on theta through sigma(theta)"

  # (sum a_i s_i)^2 expands over ORDERED pairs; this crate counts each edge once, so J_ij must be
  # -2 a_i a_j, not -a_i a_j. The wrong factor ranks every state identically and picks the same
  # ground state, so no solver notices -- only the absolute energy is off, by exactly half.
  "src/partition.rs|b.couple(i, j, -2.0 * self.a[i] as f64 * self.a[j] as f64);|b.couple(i, j, -1.0 * self.a[i] as f64 * self.a[j] as f64);|partition::tests::ising_energy_equals_the_squared_discrepancy_on_every_state|the coupling written exactly as the specification states it (J_ij = -a_i a_j), which halves the energy and leaves every ranking identical"

  # A reduction's penalty inequality is its whole correctness argument. Note the measured spec
  # error: Lucas's TSP condition A > B max(W) is not tight -- the exact critical penalty on a
  # 4-city instance is 4.5 against a stated 6.0, and the minimum violation of an infeasible state
  # is 2, never 1, so A > B T*/2 is provable.
  "src/npising.rs|b.bias(v, -(self.lin[v] / 2.0 + incident[v]));|b.bias(v, -(self.lin[v] / 2.0));|npising::tests::the_ising_energy_is_the_lucas_hamiltonian_at_every_state|drop the incident-coupling quarter from the spin bias in the x=(1+s)/2 substitution"

  # Doubling the cost exponent leaves every layer exactly unitary -- it is still the exponential of
  # a Hermitian operator -- so the unitarity test the spec called the sharp one is BLIND to it, and
  # so is the published-sequence test, because gamma' = 2 gamma is a reparametrisation of the same
  # family. Eight of nine tests stay green. Only the p = 1 closed form sees it.
  "src/qaoa.rs|let (s, c) = (gamma * diag[z]).sin_cos();|let (s, c) = (2.0 * gamma * diag[z]).sin_cos();|qaoa::tests::p1_ring_matches_the_wang_rieffel_closed_form_and_the_triangle_refutes_it|a stray factor of two in the cost exponent: e^{-i gamma C} becomes e^{-i 2 gamma C}"

  # Kac-Ward's turning angles enter through half-angles in the phase factor. Halving them is the
  # one-character error the derivation invites, and 7 of 14 tests see it -- the outer-face test,
  # the one that looks most likely to catch a phase error, is one of the 7 that does not.
  "src/pfaffian.rs|let turn = 0.5 * (wedge - PI);|let turn = wedge - PI;|pfaffian::tests::log_z_matches_exact_elimination_on_planar_graphs|drop the half-angle in the Kac-Ward phase exp(i alpha / 2)"

  # The leapfrog momentum half-step. Note the resonance the same agent hit: at eps = 1, a = 1, L = 3
  # the leapfrog map is a rotation through pi/3, three steps are half a period, and from q = 0 the
  # chain NEVER LEAVES -- variance 0.00000, dH exactly 0, acceptance exactly 1, identically in the
  # corrected and uncorrected arms. A fixture that is a fixed point measures nothing.
  "src/hmc.rs|let half = 0.5 * eps;|let half = eps;|hmc::tests::the_leapfrog_map_conserves_the_closed_form_shadow_hamiltonian|leapfrog's half-kick becomes a full kick (the halves dropped from both momentum updates)"

  # The Lyapunov descent needs a symmetric T with zero diagonal; the update reads T_ij with the
  # indices swapped. Measured alongside: forward Euler preserves descent only up to dt ~ 0.8, with
  # a sharp boundary (worst rise 1.4e-15 at 0.8, 2.2e-1 at 1.0).
  "src/analog.rs|(term(v) + term(1.0 - v)) / (2.0 * gain)|(term(v) + term(1.0 - v)) / gain|analog::tests::the_integral_term_matches_independent_quadrature|drop the factor of two in the closed form for the integral of g-inverse"

  # A port from cluster.rs carries the binary Fortuin-Kasteleyn factor of two into the q-state bond
  # probability. The sampler still runs, still binarises nothing, still converges -- to the wrong
  # temperature. Enumeration over all q^n states is what sees it.
  "src/potts.rs|if d <= 0.0 { 0.0 } else { 1.0 - (-self.beta * d).exp() }|if d <= 0.0 { 0.0 } else { 1.0 - (-2.0 * self.beta * d).exp() }|potts::tests::cluster_updates_reproduce_exhaustive_enumeration_oracle_enumerate|cluster bond probability carries the binary factor of two (the exact mistake a port from cluster.rs would make)"

  # Two-site DMRG's truncation keeps the largest singular values. Keeping the smallest still yields
  # a normalised MPS, a monotone-looking energy trace, and a number. The same agent measured that
  # the energy trace lies about convergence by exactly the size of its own error (3.8e-6 wrong,
  # 3.5e-6 spread), so the trace settling is not evidence of anything.
  "src/mps.rs|put(2, s, s, 1, -self.j * z_of(s));|put(2, s, s, 1, self.j * z_of(s));|mps::tests::the_mpo_reproduces_the_dense_hamiltonian_entry_for_entry|MPO ZZ bond term sign flipped: the -J Z half of the two-site term becomes +J Z, i.e. the crate's E = -J s s convention silently inverted for the coupling only"

  # The collapse's x-axis is (T - T_c) L^{1/nu}. Note what the same agent measured and the spec got
  # wrong: at any size this crate can compute EXACTLY (L <= 7) a free fit returns nu = 0.78 and
  # beta = 0.09 against Onsager's 1 and 1/8, and it is NOT an optimiser failure -- the residual at
  # the wrong exponents is seventy times better than at the right ones on the same objective.
  "src/fss.rs|Some(1.0 - m4 / (3.0 * m2 * m2))|Some(1.0 - m4 / (2.0 * m2 * m2))|fss::tests::the_binder_cumulant_of_free_spins_is_exactly_two_over_three_n|the Binder normalisation constant: 3 -> 2 in U_4 = 1 - m4/(3 m2^2)"

  # THE BASELINE MUST LEAVE THE EPISODE OUT. The whole-batch mean is correlated with each episode's
  # own score and returns (1 - 1/N) times the gradient -- 21.4 standard errors from exact at N = 8,
  # invisible at the N >= 4000 every caller used. Found by a second REINFORCE written independently
  # in relax.rs on the same distribution; the row names the test that measured it.
  "src/program.rs|                if episodes > 1 { (total - losses[e]) / (episodes - 1) as f64 } else { 0.0 };|                total / episodes as f64;|relax::tests::program_reinforce_is_the_exact_gradient_and_the_batch_mean_shrink_is_gone|a REINFORCE baseline that includes the episode it is subtracted from"

  # "EXACT" MEANS THE ERROR TERM IS EXACTLY ZERO, not small. Each step's error term is the exact
  # rounding error of that step, so a tolerance here is a decision to charge no guard on sums that
  # DID round -- the unsound direction. The fixture whose additions round by 2^-54 is what sees it.
  "src/round.rs|        exact &= e == 0.0;|        exact &= e.abs() <= f64::EPSILON;|round::tests::a_sum_that_plain_addition_rounds_upward_is_bracketed|a rounding error under one epsilon counted as no rounding"

  # A READ IS ONE NODE LEAVING THE CHIP, so a kept state costs n of them. Billing one per state
  # is the natural slip when the thing being counted is "draws", and it makes a cluster chain
  # n times cheaper to read out than a Gibbs chain reading the same states. The cross-module test
  # asserts the two bill the SAME reads for the same draws, and this is the row that proves it.
  "src/cluster.rs|                l.reads += self.graph.n as u64;|                l.reads += 1;|cluster::tests::a_cluster_chain_and_a_gibbs_chain_bill_the_same_reads_for_the_same_draws|a cluster chain that bills one read per state instead of one per node"


  # THE DESCENT MUST START AT THE HIGHEST POWER OF TWO BELOW n. Starting at one, it can only ever
  # test index 1, so every target past the first site's mass lands on site 0 or 1 -- a sampler that
  # still runs, still accepts and rejects, and proposes from two sites. The midpoint identity
  # against the linear scan is the test that sees it, deterministically.
  "src/informed.rs|        let mut stride = 1usize << (usize::BITS - 1 - n.leading_zeros());|        let mut stride = 1usize;|informed::tests::the_fenwick_selection_agrees_with_the_linear_scan_at_every_target|a Fenwick descent that starts at stride one and never looks past the second site"

  # The same descent, restated in this module. A path built from sites 0 and 1 only is a chain
  # that leaves the Boltzmann distribution invariant on the states it can reach and reaches almost
  # none of them; the enumeration TV test is what sees it.
  "src/multiflip.rs|    let mut stride = 1usize << (usize::BITS - 1 - n.leading_zeros());|    let mut stride = 1usize;|multiflip::tests::the_path_chain_samples_the_boltzmann_distribution|a path whose every flip is drawn from the first two sites"


  # P = P_last ... P_first, so applying P to a FUNCTION over states runs the classes in reverse.
  # Forward order builds the transpose composition: still a stochastic operator, still leaves pi
  # invariant (each factor does), and is the wrong kernel. Only the dense operator assembled
  # class by class in the test, by different code, tells them apart.
  "src/autocorr.rs|            for class in g.classes.iter().rev() {|            for class in g.classes.iter() {|autocorr::tests::the_matrix_free_sweep_matches_a_dense_operator_built_class_by_class|the sweep operator composed in the wrong class order"


  # THE CROSS-CHECK MUST BE ABLE TO FIRE. A threshold of 200x is a check that is wired, compiles,
  # and never speaks: every certificate keeps Sokal's tau and the ESS it inflates. The cold 3x3
  # chain, whose exact tau the oracle puts in the tens against a single-digit window, is what
  # requires the finding to appear -- and the hot half of the same test is what stops the fix from
  # being "always fire".
  "src/certify.rs|    let truncated = t_sokal.is_finite() && t_batch.is_finite() && t_batch > 2.0 * t_sokal;|    let truncated = t_sokal.is_finite() && t_batch.is_finite() && t_batch > 200.0 * t_sokal;|certify::tests::the_certificate_reports_a_truncated_window_and_carries_the_larger_tau|a truncation cross-check whose threshold nothing real can reach"


  # A SWEEP ON FUNCTIONS RUNS THE SITES BACKWARDS. P = P_0 P_1 ... P_{n-1} on distributions, so
  # (P v) applies the last site first; forward order builds the sweep for the reversed site order,
  # which is still a stochastic operator with pi invariant and is not the kernel Dtm::sample runs.
  # Only the adjoint identity against apply_distribution -- written forward, as mass moves --
  # tells the two apart.
  "src/autocorr.rs|            for i in (0..g.n).rev() {|            for i in 0..g.n {|autocorr::tests::pushing_mass_forward_is_the_adjoint_of_pulling_functions_back|the sequential sweep applied to functions in forward site order"


  # THE BUDGET MUST NOT REFUSE WHAT IT CAN DO. A budget a thousand times too small refuses the
  # 4,096-trajectory case the in-budget test runs, and that test panics at once. The OTHER
  # direction -- dropping the budget so nv = 9, T = 4 proceeds -- is a mutation this suite cannot
  # witness: the should-panic test then waits for a 3.5e13-step enumeration that never returns.
  # That row was written first, and it HUNG the suite for half an hour before it was understood;
  # a mutation that turns a refusal into a hang is invisible to `cargo test`, which is why the
  # check now carries a watchdog and why this row is the one that can be observed.
  # (The comparison sits on its own line, without a closure, because a row cannot carry the
  # suite's `|` separator -- the first version of this row did, inside `|c| c <= ...`, and would
  # have been read as a shifted set of fields.)
  "src/dtm.rs|        let within_budget = count <= MAX_NLL_TRAJECTORIES;|        let within_budget = count < 4096;|dtm::tests::exact_nll_runs_inside_its_budget|an exact_nll budget so small it refuses what it can do"


  # THE KERNEL IS THE RTL'S ARITHMETIC OR IT IS NOTHING. Reading the ROM at half the stride is a
  # kernel that still runs, still converges to a law, and is not the fabric's: the cycle-exact
  # emulator's own histogram is what says so, which is the only witness that ties the operator
  # to the hardware rather than to a description of it.
  # Row 118 originally mutated the fixed-stride address line, which 5105303 refactored into the
  # Quantised arithmetic; the 127-row run of 2026-09-13 reported MUTATION DID NOT APPLY, which is
  # the suite doing its job. Retargeted to the ROM centre, the one part of the address arithmetic
  # no other row touches.
  "src/autocorr.rs|            let arg = (addr as f64 + 0.5) * stride - 8.0;|            let arg = addr as f64 * stride - 8.0;|autocorr::tests::the_rom_is_read_at_the_cell_centre|a fabric kernel reading the sigmoid ROM at the cell edge"


  # THE SHIPPED PRECISION MUST BE ONE POINT OF THE KNOB, EXACTLY. One bit off in the address
  # shift is a kernel that runs and converges and is not FixedFabric's; the bit-for-bit equality
  # of Quantised {8, 10, 16} with the fabric kernel is what sees it, and it is the anchor that ties
  # every other point of the precision sweep to the hardware that was metered.
  "src/autocorr.rs|                offset >> (frac_bits + 4 - lut_bits)|                offset >> (frac_bits + 3 - lut_bits)|autocorr::tests::the_quantised_kernel_reduces_to_the_fabric_and_improves_with_bits|a quantised kernel whose ROM address is off by one bit of stride"
  # Rows 120-122 (2026-09-13): the DIRECT oracle. The lag sum and the mass-pushing iteration are
  # both mixing-time computations and were cut off by their budgets at beta 1.5 on 12 spins; the
  # dense solves replace them. Each row breaks one identity the solve rests on.
  "src/autocorr.rs|    let tau = total / c0 - 0.5;|    let tau = total / c0;|autocorr::tests::the_fundamental_matrix_tau_agrees_with_the_lag_sum_and_needs_no_lags|a fundamental-matrix tau that forgets the half"
  "src/autocorr.rs|        a[(m - 1) * m + c] = 1.0;|        a[(m - 1) * m + c] = 0.0;|autocorr::tests::the_direct_solve_reproduces_boltzmann_for_gibbs_and_the_iterated_law_for_the_fabric|a stationary solve without its normalisation"
  # The sign of the rank-one term is NOT a mutation: pi^T z = 0 for the solution, so I - P + c pi 1^T
  # has the same solution for every c != 0 and a test cannot see the sign. Tried, STILL GREEN,
  # and correctly so -- an equivalent mutant. Dropping the term is the defect: I - P is singular.
  "src/autocorr.rs|            a[x * m + y] = delta - p + pi[y];|            a[x * m + y] = delta - p;|autocorr::tests::the_fundamental_matrix_tau_agrees_with_the_lag_sum_and_needs_no_lags|a fundamental matrix without its rank-one term, which is singular"
  "src/autocorr.rs|    Ok(trace - 1.0)|    Ok(trace)|autocorr::tests::kemenys_constant_has_its_closed_forms_on_memoryless_kernels|a Kemeny constant that counts the stationary mode"
  "src/autocorr.rs|            let levels = 2f64.powi(prob_bits as i32);|            let levels = f64::from(1u32 << (prob_bits % 32));|autocorr::tests::the_quantised_kernel_at_full_precision_is_the_exact_kernel|a comparator width computed in 32-bit integers, which wraps at 32 bits"
  "src/autocorr.rs|                    let q_prev = p_site(g, beta, kernel, i, &s);|                    let q_prev = p_site(g, 1.0, kernel, i, &s);|autocorr::tests::the_synchronous_kernel_leaves_perettos_law_invariant_and_it_is_not_boltzmann|a synchronous sweep that samples at unit temperature"
  "src/autocorr.rs|        let mut acc = beta * hx;|        let mut acc = 0.0 * hx;|autocorr::tests::the_synchronous_kernel_leaves_perettos_law_invariant_and_it_is_not_boltzmann|a Peretto law without its field term"
  "src/autocorr.rs|            let z = ((beta * g.field(i, s)).tanh() + xi * f64::from(s[i])) / eta;|            let z = (beta * g.field(i, s)).tanh() / eta;|autocorr::tests::the_pimi_kernel_has_its_single_site_closed_form_and_its_inertia_matters|a PIMI rule without its inertia term"
  "src/autocorr.rs|            let s_j = if stale { x_j } else { y_j };|            let s_j = if stale { y_j } else { x_j };|autocorr::tests::the_stale_read_kernel_is_bracketed_by_its_two_closed_forms|a stale read that returns the fresh value"
  # Rows 129-130 (2026-09-13): the FFT. Its oracle is an independently written O(N^2) DFT and the
  # direct lag-by-lag sum; each row breaks the one property the route rests on.
  "src/fft.rs|        let ang = if inverse { TAU / len as f64 } else { -TAU / len as f64 };|        let ang = if inverse { TAU / len as f64 } else { TAU / len as f64 };|fft::tests::the_transform_matches_the_direct_dft_and_inverts|a forward transform with the inverse's twiddle sign"
  "src/fft.rs|    let m = (2 * n).next_power_of_two();|    let m = n.next_power_of_two();|fft::tests::the_autocovariance_matches_the_direct_sum_lag_by_lag_and_sokal_agrees|an autocovariance without zero padding, so the lags wrap"
  "src/certify.rs|            Some(cov) => cov[k] / var,|            Some(cov) => cov[k],|certify::tests::the_fft_path_and_the_direct_sum_agree|a transform-path autocorrelation left unnormalised"
  "src/autocorr.rs|        Kernel::SiteSpread { seed, spread } => p_up(g.field(i, s), beta * site_factor(seed, spread, i)),|        Kernel::SiteSpread { seed, spread } => p_up(g.field(i, s), beta * site_factor(seed, 0.0 * spread, i)),|autocorr::tests::a_site_temperature_spread_reduces_to_the_sweep_at_zero_and_to_a_product_on_free_sites|a site temperature spread that is never applied"
  "src/autocorr.rs|            sigma += fwd * (fwd / bwd).ln();|            sigma += fwd * fwd.ln();|autocorr::tests::entropy_production_is_zero_for_reversible_kernels_and_positive_for_a_correct_fixed_order_sweep|an entropy production that never looks at the reverse transition"
  "src/autocorr.rs|    let refined = apply_distribution(g, beta, kernel, &clamped);|    let refined = clamped.clone();|autocorr::tests::the_solved_law_has_no_zero_entries_on_a_cold_grid_and_the_sweep_produces_finite_entropy|a solved law whose small entries are the solve's noise, clamped"
  "src/reduce.rs|        Some(v) if v.is_finite() && v > 0.0 => v,|        Some(v) if v.is_finite() && v > 0.0 => v.max(default),|reduce::tests::a_chosen_penalty_is_written_and_a_weak_one_lets_an_ancilla_break|a chosen penalty that is never allowed below the default"
  "src/model.rs|                let scale = stage.penalties.domain_wall;|                let scale = 1.0;|model::tests::a_scaled_codeword_penalty_is_applied_and_a_ramp_ends_where_it_says|a penalty ramp that every stage ignores, as every solver did until 2026-09-13"
  "src/rhat.rs|        out.push(c[half..2 * half].to_vec());|        out.push(c[..half].to_vec());|rhat::tests::a_trend_inside_every_chain_is_caught_by_splitting|an R-hat whose second half is the first half again, so a trend is invisible"
  "src/rhat.rs|            folded.push((x - med).abs());|            folded.push(x - med);|rhat::tests::a_wrong_spread_with_the_right_centre_is_caught_by_folding|a folded R-hat that does not fold"
  "src/landauer.rs|    BOLTZMANN_CONSTANT * temperature_k * core::f64::consts::LN_2|    BOLTZMANN_CONSTANT * temperature_k|landauer::tests::the_floor_at_room_temperature_and_the_devices_above_it|a Landauer bound that forgets the ln 2"
  "src/landauer.rs|            integral += fwd * (-a).exp();|            integral += fwd * a.exp();|landauer::tests::the_fluctuation_theorems_hold_exactly_on_a_reversible_support_kernel|an integral fluctuation theorem with the sign of the entropy flipped"
  "src/restart.rs|        steps += 1.0 - p;|        steps += 1.0;|restart::tests::a_memoryless_solver_gains_nothing_from_any_cutoff|an expected work that bills every attempt its full cutoff"
  "src/restart.rs|            return 1u64 << (k - 1);|            return 1u64 << k;|restart::tests::lubys_sequence_is_the_published_one_and_its_blocks_sum_as_they_must|a Luby sequence whose block ends are twice too long"

  # The free-fermion oracle. Pfeuty's single-particle energy is checked against a dense
  # Hamiltonian on 4, 6 and 8 spins; drop its cosine cross term and the spectrum is flat, which
  # the ordered and disordered cases both see. The momenta belong to the even-parity sector
  # (antiperiodic); the periodic ones give a finite-size coefficient of +1.047 instead of -pi/6,
  # and only the Casimir test notices, because both decoupled limits are blind to the sector.
  "src/freefermion.rs|2.0 * j * gamma * k.cos()).sqrt()|0.0 * j * gamma * k.cos()).sqrt()|freefermion::tests::pfeutys_closed_form_is_the_exact_ground_state_of_the_dense_hamiltonian|a single-particle energy without its cosine cross term"
  "src/freefermion.rs|PI * (2.0 * m as f64 + 1.0) / n as f64|PI * (2.0 * m as f64) / n as f64|freefermion::tests::the_critical_chains_finite_size_correction_is_the_casimir_term_of_c_one_half|momenta from the wrong parity sector"
  # Katsura's trace is four products; the odd sector's sinh product enters with a MINUS, which is
  # the parity projection (1 - P)/2. Add it instead and Z at J = 0 is (2cosh)^N + (2sinh)^N, not
  # (2cosh)^N; the whole-spectrum Jacobi test sees it in every phase.
  "src/freefermion.rs|    let mut odd_sign = -1.0;|    let mut odd_sign = 1.0;|freefermion::tests::katsuras_finite_temperature_solution_is_the_whole_dense_spectrum|a periodic sector whose parity projection adds instead of subtracts"

  # The annealer's access cost. The initialisation overhead is half the fixed cost; forget it and
  # a one-read submission is a third cheaper than the published model says. The reads that amortise
  # the fixed cost are a ceiling; a floor under-counts by one, which the pinned 208 sees.
  "src/access.rs|        qpu.programming_us + self.overhead_us|        qpu.programming_us|access::tests::one_read_costs_a_third_of_a_kilojoule_before_it_anneals|an access time that forgets the initialisation overhead"
  "src/access.rs|            .ceil()|            .floor()|access::tests::reads_amortise_the_fixed_cost_as_the_closed_form_says|an amortisation count rounded the wrong way"

  # The p-dit. The Gumbel ROM is -ln(-ln u); drop the outer sign and the unit's exact law is no
  # longer the softmax, which the ten-bit total-variation bound sees. The exact law's bound for a
  # rival state is score - field_b; add instead of subtract and the law is wrong for any
  # asymmetric fields. The encoded row's valid mass is a sum over decodable states; zero it and the
  # comparison test's "some mass is valid" assertion goes. The lowering's constant carries
  # -J c_i c_j per edge and state; flip it and the valid-codeword energies no longer match. The
  # emulator's ROM address is the top rom_bits of a draw; one bit wider indexes past the ROM. The
  # RTL chains its q draws; feed every draw the same seed word and the icarus gate sees the trace
  # diverge from the emulator.
  "src/pdit.rs|            let g = -(-u.ln()).ln();|            let g = (-u.ln()).ln();|pdit::tests::the_rom_units_exact_law_is_the_softmax_until_the_gumbel_range_ends|a Gumbel ROM with its outer sign dropped"
  "src/pdit.rs|                let bound = score - fields_q[b];|                let bound = score + fields_q[b];|pdit::tests::the_rom_units_exact_law_is_the_softmax_until_the_gumbel_range_ends|a rival's bound with the wrong sign"
  "src/pdit.rs|            valid_mass += pi[x];|            valid_mass += 0.0;|pdit::tests::a_native_p_trit_reaches_stationarity_in_fewer_draws_than_its_encodings|an encoded row that reports no valid mass"
  "src/pdit.rs|            constant -= w * ci * cj;|            constant += w * ci * cj;|pdit::tests::the_lowerings_agree_with_the_potts_energy_on_every_valid_codeword|a lowering constant with its sign flipped"
  "src/pdit.rs|        let shift = 32 - self.rom_bits;|        let shift = 31 - self.rom_bits;|pdit::tests::the_fixed_point_p_trit_fabric_samples_the_potts_law|a ROM address one bit too wide"
  "src/pdit.rs|                    format!(\"d{i}_{}\", a - 1)|                    format!(\"rng[{i}]\")|pdit::tests::the_p_trit_rtl_matches_the_emulator_bit_exact|an RTL whose q draws are all the first draw"

  # The p-bit emitter wrote a negative bias as 32'sd-154, which is not Verilog; the gate never
  # saw it because every lattice it ran on was unbiased. Now it carries biases of both signs, and
  # an emitter that drops the negated literal's sign puts the wrong field in the netlist.
  "src/hdl.rs|                format!(\"-32'sd{}\", -bias)|                format!(\"32'sd{}\", -bias)|hdl::tests::verilog_matches_emulator_bit_exact|a negative bias emitted with its sign dropped"

  # The cumulative p-dit. Its exponential ROM is e^-gap; double the exponent and the unit samples
  # at twice the temperature it was quantised for, which the exact-law bound sees. The emulator's
  # threshold is (u Z) >> 16; shift by one less and the threshold overruns the cumulative sum, so
  # the walk lands on the last state too often -- the histogram against the exact law sees it.
  # The RTL's threshold is the same shift; take one bit more of the product and the netlist and
  # the emulator part company, which the icarus gate sees.
  "src/pdit.rs|            (top * (-gap).exp()).round() as u32|            (top * (-2.0 * gap).exp()).round() as u32|pdit::tests::the_cumulative_units_exact_law_is_the_softmax_until_the_weight_width_ends|an exponential ROM at twice the temperature"
  "src/pdit.rs|        let threshold = (u * z) >> WEIGHT_BITS;|        let threshold = (u * z) >> (WEIGHT_BITS - 1);|pdit::tests::the_cumulative_p_trit_fabric_samples_the_potts_law|a threshold that overruns the cumulative sum"
  "src/pdit.rs|p{i}[47:16]|p{i}[47:15]|pdit::tests::the_cumulative_p_trit_rtl_matches_the_emulator_bit_exact|an RTL threshold one bit too wide"

  # Decompositions. The singular values are the column norms after the Jacobi rotations; square
  # them and the prescribed spectrum comes back squared. The HOSVD's core is the tensor contracted with each
  # factor's transpose; scale a transpose and the reconstruction is off by that scale. The
  # randomized range finder's sketch is k + p columns wide; drop the oversampling and the top
  # singular values are no longer recovered to a millionth.
  "src/decomp.rs|            s[t] = norms[j];|            s[t] = norms[j] * norms[j];|decomp::tests::the_svd_reconstructs_and_recovers_a_prescribed_spectrum|singular values squared"
  "src/decomp.rs|                ut[t * rows + i] = factor[i * r + t];|                ut[t * rows + i] = factor[i * r + t] * 0.5;|decomp::tests::the_hosvd_reconstructs_and_its_core_is_all_orthogonal|a Tucker core contracted with half a factor"
  "src/decomp.rs|    let l = (k + oversample).min(rows).min(cols);|    let l = k.min(rows).min(cols);|decomp::tests::the_randomized_svd_matches_the_full_one_on_a_decaying_spectrum|a range finder with no oversampling"

  # Sourlas codes. The Nishimori temperature is half the log-likelihood ratio; double it and the
  # identity that is zero only on the Nishimori line is no longer zero. The channel average weights
  # a pattern by p^flips (1-p)^rest; swap the two and the same identity fails. The expected error
  # counts a marginal of the wrong sign; count the right sign instead and the Nishimori minimum
  # is a maximum. The Ising posterior
  # carries the received coupling itself; negate it and the Gibbs kernel decodes the wrong model.
  "src/sourlas.rs|0.5 * ((1.0 - p) / p).ln()|((1.0 - p) / p).ln()|sourlas::tests::nishimoris_identity_holds_exactly_on_the_nishimori_line|a Nishimori temperature twice too hot"
  "src/sourlas.rs|let prob = p.powi(count as i32) * (1.0 - p).powi((m - count as usize) as i32);|let prob = (1.0 - p).powi(count as i32) * p.powi((m - count as usize) as i32);|sourlas::tests::nishimoris_identity_holds_exactly_on_the_nishimori_line|a channel that weights flips as keeps"
  "src/sourlas.rs|        } else if agree < 0.0 {|        } else if agree > 0.0 {|sourlas::tests::the_nishimori_temperature_minimises_the_expected_bit_error|a decoder that reads the marginal's sign backwards"
  "src/sourlas.rs|2 => b.couple(s[0], s[1], f64::from(j)),|2 => b.couple(s[0], s[1], -f64::from(j)),|sourlas::tests::a_gibbs_kernel_decodes_as_the_posterior_and_the_fabric_agrees_on_a_noisy_channel|an Ising posterior with its couplings negated"

  # GF(2). A solve is inconsistent when the augmented column pivots; invert the test and it
  # answers inconsistent systems and refuses solvable ones. A nullspace vector sets a pivot bit
  # where the reduced row has a one in the free column; invert that and the matrix no longer
  # annihilates it. The reduction swaps the pivot row into place; skip the swap and the row it
  # eliminates with lacks the pivot, which the brute-force nullspace sees.
  "src/gf2.rs|        if pivots.contains(&self.cols) {|        if !pivots.contains(&self.cols) {|gf2::tests::solve_agrees_with_brute_force_on_consistency_and_solves_when_it_can|a solver that inverts consistency"
  "src/gf2.rs|                    if r.get(i, f) {|                    if !r.get(i, f) {|gf2::tests::the_nullspace_is_exactly_what_brute_force_finds|a nullspace built from the complement"
  "src/gf2.rs|            m.swap_rows(row, p);|            m.swap_rows(row, row);|gf2::tests::the_nullspace_is_exactly_what_brute_force_finds|a reduction that never brings the pivot up"

  # Syndrome decoding. Belief propagation flips a check message's sign on an odd syndrome; flip it the other
  # way and BP is no longer exact on the tree. The relaxed factor's energy is -gamma J prod;
  # drop the sign and the relaxation converges to the wrong posterior. The hard marginal counts
  # a flipped bit as -1; count it as +1 and the marginals invert.
  "src/syndrome.rs|let sign = if syndrome[mu] == 1 { -1.0 } else { 1.0 };|let sign = if syndrome[mu] == 1 { 1.0 } else { -1.0 };|syndrome::tests::belief_propagation_is_exact_on_a_tree_and_the_hard_posterior_is_the_relaxed_limit|check messages with the syndrome's sign reversed"
  "src/syndrome.rs|            *entry = -gamma * j * prod;|            *entry = gamma * j * prod;|syndrome::tests::belief_propagation_is_exact_on_a_tree_and_the_hard_posterior_is_the_relaxed_limit|a relaxed check that rewards the wrong parity"
  "src/syndrome.rs|marg[i] += w * if b == 1 { -1.0 } else { 1.0 };|marg[i] += w * if b == 1 { 1.0 } else { -1.0 };|syndrome::tests::belief_propagation_is_exact_on_a_tree_and_the_hard_posterior_is_the_relaxed_limit|hard marginals with the flip sign inverted"

  # The p-dit's AXI shell counts a sweep when the phase wraps; count two and the fabric stops at
  # half the target while the host reads twice the sweeps, which the shell gate sees.
  "src/pdit.rs|done_sweeps <= done_sweeps + 32'd1;|done_sweeps <= done_sweeps + 32'd2;|pdit::tests::the_p_dit_axi_shell_runs_the_fabric_and_reads_back_what_the_emulator_reaches|a shell that counts every sweep twice"
  # The p-bit shell's counter, after the same fix: the gate now runs hot enough that the state
  # changes every sweep, so a counter that is off by one phase or by a factor of two shows.
  "src/hdl.rs|      if (phase) done_sweeps <= done_sweeps + 32'd1;|      if (phase) done_sweeps <= done_sweeps + 32'd2;|hdl::tests::axi_shell_runs_the_fabric_and_reads_back_what_the_emulator_reaches|a p-bit shell that counts every sweep twice"
  # And the defect itself: count on the other phase and the fabric runs one class past the
  # target, which is the half-sweep overrun the frozen gate could not see.
  "src/hdl.rs|      if (phase) done_sweeps <= done_sweeps + 32'd1;|      if (!phase) done_sweeps <= done_sweeps + 32'd1;|hdl::tests::axi_shell_runs_the_fabric_and_reads_back_what_the_emulator_reaches|a p-bit shell that counts on the wrong phase"

  # The silicon router's first row. An interconnect column is interrupted by a clock row once per
  # clock region, and the wires continue straight through it -- 223 pairs in each vertical
  # direction on an xc7a100t. Treat that row as a wall and every coupling inside a region still
  # routes while every coupling across one fails, which is how this shipped: 62 of 63 couplings
  # routed and the one failure looked like a hard case.
  "silicon/src/route.rs|pub const CLOCK_ROW_PREFIXES: [&str; 2] = [\"HCLK_L\", \"HCLK_R\"];|pub const CLOCK_ROW_PREFIXES: [&str; 2] = [\"HCLK_WALL\", \"HCLK_R\"];|route::tests::a_route_crosses_the_horizontal_clock_row|a router that treats the clock row as a wall|ferrotherm-silicon"
  # A .bit header field declares a length that INCLUDES its NUL terminator. Drop the NUL from the
  # count and the container still parses -- the reader simply returns a string one byte short and
  # every following field lands one byte early -- so only a byte-for-byte comparison against the
  # container the parser was always tested against catches it.
  "silicon/src/bitstream.rs|        let len = text.len() + 1;|        let len = text.len();|bitstream::tests::the_writer_emits_the_container_the_parser_was_tested_against|a header field whose length forgets its terminator|ferrotherm-silicon"

  # A cross-check that reports success on no evidence is the failure it was written to prevent:
  # with an empty listing nothing was compared, so `disagreed.is_empty()` is true and means
  # nothing. Drop the `agreed > 0` half and an unverified grid passes as verified.
  "silicon/src/logic_location.rs|        self.agreed > 0 && self.disagreed.is_empty()|        self.disagreed.is_empty()|logic_location::tests::an_empty_listing_verifies_nothing|a cross-check that verifies on no evidence|ferrotherm-silicon"

  # Two nets on one wire is the smallest contention there is, and the one a chain of couplings
  # actually produces. Report only three or more and the pair -- two drivers on one conductor --
  # goes out to the device, which answers it with current rather than an error.
  "silicon/src/route.rs|        if nets.len() > 1 {|        if nets.len() > 2 {|route::tests::two_nets_driving_one_wire_are_reported|a contention check that needs three drivers|ferrotherm-silicon"

  # The parity bit is inside the count it corrects. Count first and write second, without clearing
  # the bit that is already there, and the answer alternates: right on a fresh frame, wrong on the
  # second pass over the same one. Both passes return a plausible bool and the frame still loads.
  "silicon/src/ecc.rs|    frame[word] &= !mask;|    frame[word] &= u32::MAX;|ecc::tests::setting_parity_is_idempotent|a parity write that counts the bit it is about to write|ferrotherm-silicon"

  # A survey of no frames has nothing odd in it. Drop the clause that requires frames to have been
  # counted and the verdict reads VENDOR-SHAPED for a file with no configuration stream at all --
  # the same shape as a cross-check that verifies because it compared nothing.
  "silicon/src/ecc.rs|        self.frames > 0 && self.odd == 0 && self.remainder == 0|        self.odd == 0 && self.remainder == 0|ecc::tests::an_empty_survey_is_not_vendor_shaped|a parity verdict that passes on an empty stream|ferrotherm-silicon"

  # UltraScale+ widened the frame address's minor field to eight bits, and its block-RAM content
  # columns are 256 frames deep. Carry the 7-series seven-bit mask across and the top half of every
  # wide column folds silently onto the bottom half: every address still decodes, every frame still
  # writes, and half of them land somewhere else.
  "silicon/src/usplus.rs|            minor: (far & 0xFF) as u8,|            minor: (far & 0x7F) as u8,|usplus::tests::the_minor_field_is_eight_bits|a 7-series minor mask applied to UltraScale+|ferrotherm-silicon"

  # The boot header of an UltraScale+ image contains the sync word as its bus-width pattern. Take
  # the first match and the reader parses the header as configuration, finds no frames, and reports
  # an empty design for a file that holds twenty thousand frames.
  "silicon/src/usplus.rs|        if config.get(i + 1) == Some(&NOOP) {|        if config.get(i + 1).is_some() {|usplus::tests::the_first_sync_word_is_not_the_configuration_stream|a sync search that stops at the boot header|ferrotherm-silicon"

  # A frame readback streams one frame of pipeline contents before the frame that was asked for.
  # Forget it and every frame is read one address late -- with real bits, from the real device, and
  # nothing anywhere that looks wrong. The pad frame in the fixture is deliberately not zero.
  "silicon/src/capture.rs|pub const PAD_FRAMES: usize = 1;|pub const PAD_FRAMES: usize = 0;|capture::tests::the_pad_frame_is_dropped_and_a_short_read_refused|a readback that keeps the pipeline frame|ferrotherm-silicon"

  # A JTAG shift that reads returns a byte for every byte sent. Send the whole shift before reading
  # any of it and the FTDI's return buffer fills, its engine stalls, the host's write never
  # completes, and nothing errors: on 2026-09-19 a 36-frame readback of a real XC7A100T sat in a
  # blocking write for twenty-nine minutes. The plan is a pure function so that this can be a row.
  "silicon/src/mpsse.rs|    for chunk in tx[..n - 1].chunks(RETURN_BUFFER_SAFE) {|    for chunk in tx[..n - 1].chunks(65536) {|mpsse::tests::no_step_asks_the_chip_to_hold_more_than_it_can|a captured shift sent whole into a 4 KB buffer|ferrotherm-silicon"

  # The other half of the same deadlock: cut the shift but never collect the pieces' replies.
  "silicon/src/mpsse.rs|        steps.push(Step { cmd, reads: chunk.len() });|        steps.push(Step { cmd, reads: 0 });|mpsse::tests::no_step_asks_the_chip_to_hold_more_than_it_can|a step that writes and never collects its reply|ferrotherm-silicon"

  # Cutting a shift into pieces must not re-enter Shift-DR for each one. Do that and every piece
  # after the first passes through Update-DR and Capture-DR: the register is reloaded mid-read and
  # the bytes that come back are real, plausible, and from the wrong place.
  "silicon/src/mpsse.rs|        let mut cmd = core::mem::take(&mut prefix);|        let mut cmd = prefix.clone();|mpsse::tests::the_plan_shifts_exactly_the_bytes_it_was_given|a shift that re-enters Shift-DR for every piece|ferrotherm-silicon"

  # The last bit of a shift does not travel as data; it rides the TMS clock that leaves Shift-DR.
  # THIS MUTANT SURVIVED ITS FIRST RUN: the fixture's last byte was 0x3A, top bit clear, so a plan
  # that lost the bit walked back to the same payload. The fixture now ends in 0xC3.
  "silicon/src/mpsse.rs|((last >> 7) << 7)|((last >> 7) << 6)|mpsse::tests::the_plan_shifts_exactly_the_bytes_it_was_given|a final data bit placed where the chip ignores it|ferrotherm-silicon"

  # The metered read and the metered flip are RE-DERIVED from the sensor logs committed beside
  # them, so a constant edited away from its evidence fails against the evidence. Two rows because
  # the two constants come from two logs.
  "src/ledger.rs|    e_read: 5.8126e-10,|    e_read: 5.8126e-12,|ledger::tests::the_metered_read_is_rederived_from_its_own_sensor_log|a read price that drifts from its sensor log"
  "src/ledger.rs|    e_sample: 9.1316e-12,|    e_sample: 9.1316e-11,|ledger::tests::the_metered_read_is_rederived_from_its_own_sensor_log|a flip price that drifts from its sensor log"

  # A spin read costs about 64 flips on the board that metered both. Price a read as a flip and a
  # run that reads its state every draw loses half its bill, which is the half the old price set
  # could not state and the reason it refused.
  "src/ledger.rs|                + charge(self.reads, p.e_read)?|                + charge(self.reads, p.e_sample)?|floors::tests::a_certified_draw_is_priced_with_its_readout_on_metered_silicon|a read priced as a flip"

  # THIS MUTANT SURVIVED ITS FIRST RUN. The two KV260 entries share a grade and an instrument, so a
  # table that handed out the older board's prices under the newer board's name passed every test:
  # nothing bound any catalogue name to the constant it names. The grade test now does.
  "src/ledger.rs|    (\"KV260_AXI_METERED\", KV260_AXI_METERED),|    (\"KV260_AXI_METERED\", KV260_MEASURED),|ledger::tests::only_the_metered_price_claims_to_be_metered|a catalogue name bound to another machine's prices"

  # THE WRITABLE FABRIC. Its contract is that a fabric programmed with a graph IS the fixed fabric
  # of that graph, state for state -- in the emulator, and in the emitted RTL driven over AXI by
  # the same bus words `program` returns. Each row breaks one link of that chain: the emulator's
  # write, the refusals that stop a half-fitting problem being written, and the RTL's decoder and
  # four-clock schedule. (Swapping the two halves of a bus word was also tried and caught; it
  # contains a pipe and cannot be a row, so the shifted-field form below stands in for it.)
  "src/writable.rs|(hi << 16) });|(hi << 12) });|writable::tests::rtl_written_over_axi_matches_the_fixed_fabric_of_what_was_written|the second weight of a pair landing four bits low in its bus word"
  "src/writable.rs|                self.core.adj[i][slot].1 = w;|                self.core.adj[i][slot].1 = -w;|writable::tests::a_programmed_fabric_is_the_fixed_fabric_of_its_target|an emulator that writes the negated weight"
  "src/writable.rs|            if nbrs != wired.as_slice() {|            if false {|writable::tests::a_weight_that_does_not_fit_is_refused_and_nothing_is_written|the same node count wired differently and accepted"
  "src/writable.rs|if (phase & sub) done_sweeps|if (phase) done_sweeps|writable::tests::rtl_written_over_axi_matches_the_fixed_fabric_of_what_was_written|a sweep counter that forgets a sweep is four clocks"
  "src/writable.rs|    if !(f64::from(WMIN)..=f64::from(WMAX)).contains(&q) {|    if !(f64::from(WMIN)..=f64::from(WMAX + 1)).contains(&q) {|writable::tests::a_weight_that_does_not_fit_is_refused_and_nothing_is_written|a weight of +8.0 accepted and wrapped to -8.0"
  "src/writable.rs|cfg_word <= awaddr_r[{word_hi}:2];|cfg_word <= awaddr_r[{word_hi}:2] ^ 1;|writable::tests::rtl_written_over_axi_matches_the_fixed_fabric_of_what_was_written|configuration words that land in the neighbouring word"

  # A slave that drops its write response before the master has accepted it. The polite testbench
  # ties `bready` high and CANNOT see this -- run against this same mutant it stays green -- and
  # on a board it is a core stalled for ever on a transaction that will never complete. Both
  # shells are held to a master that skews its channels and accepts responses late.
  "src/writable.rs|      if (bvalid_r && s_axi_bready) begin|      if (bvalid_r) begin|writable::tests::neither_shell_deadlocks_under_a_badly_behaved_master|a writable shell that drops a response nobody has accepted"
  "src/hdl.rs|      if (bvalid_r && s_axi_bready) begin|      if (bvalid_r) begin|writable::tests::neither_shell_deadlocks_under_a_badly_behaved_master|a fixed shell that drops a response nobody has accepted"

  # THE WRITABLE DEVICE. Its ladder anneals because a write touches weights and nothing else; it
  # refuses a rung its registers cannot hold BEFORE running any of the ladder; and it charges a
  # load for a new seed, because seeds are reset constants of the netlist.
  #
  # THE LAST ROW SURVIVED ITS FIRST RUN, and the defect was in the gate. An RTL that reset the chain
  # on every configuration write still reproduced the "carried" state: the run resumed at the same
  # position in every generator's stream, and eleven hot sweeps on shared random numbers made the
  # two chains COALESCE -- the mechanism `cftp` is built on, here erasing the memory the gate
  # existed to see. The gate is now two cold sweeps long and asserts against that alternative.
  "src/writable.rs|                f.core.reset(); // the shell|                // the shell|writable::tests::a_new_seed_is_a_new_netlist_and_is_charged_as_a_load|a rerun that continues where the last run stopped"
  "src/writable.rs|        if self.held_scale == Some(c) {|        if false && self.held_scale == Some(c) {|writable::tests::a_ladder_on_the_writable_fabric_anneals_and_costs_one_more_write_not_fewer|an unchanged temperature charged as a write"
  "src/writable.rs|            self.set_temperature(stage.beta)?;|            self.set_temperature(stage.beta)?; self.fabric.as_mut().expect(\"s\").core.reset();|writable::tests::a_ladder_on_the_writable_fabric_anneals_and_costs_one_more_write_not_fewer|a writable device that resets at every rung like the fixed one"
  "src/writable.rs|            if raw != 0.0 && (*v).abs() < step / 2.0 {|            if false && raw != 0.0 && (*v).abs() < step / 2.0 {|writable::tests::a_rung_the_registers_cannot_hold_refuses_the_whole_ladder_before_running_any_of_it|a coupling deleted by rounding and run anyway"
  "src/writable.rs|                let t = WritableRtl::scaled(g, stage.beta / ROM_BETA)?;|                let t = WritableRtl::scaled(g, 1.0)?;|writable::tests::a_rung_the_registers_cannot_hold_refuses_the_whole_ladder_before_running_any_of_it|a ladder refused only after its good rungs ran"
  "src/writable.rs|        if self.seed == Some(seed) {|        if self.seed.is_some() {|writable::tests::a_new_seed_is_a_new_netlist_and_is_charged_as_a_load|a new seed served by the old netlist for free"
  "src/writable.rs|          cfg_en   <= 1'b1;|          cfg_en   <= 1'b1; soft_rst <= 1'b1;|writable::tests::rtl_reprogrammed_mid_run_without_a_reset_carries_its_state|RTL that resets the chain on every configuration write"

  # THE APPLICATION LAYER. Its energy is a DERIVATION, not a model with a tunable data weight: the
  # field is the channel's log-odds and the couplings are the prior.
  #
  # ROW 1 SURVIVED ITS FIRST RUN. Replacing the whole log-odds with the constant 1.0 passed every
  # test in the module, because those tests run at p = 0.25 where ln((1-p)/p) = 1.0986 -- within
  # ten percent of the constant replacing it. The parameter, not the assertion, was the blind spot,
  # and `the_field_is_the_channels_log_odds_and_not_a_tunable_weight` is the answer to it.
  #
  # Two further mutants were run by hand and caught but cannot be rows, their text containing a
  # pipe: a per-pixel mode taken at threshold 0.4 instead of 0.5, and a bill that silently drops an
  # operation its prices could not price. Both RED. The refusal rows below are likewise written in
  # their pipe-free form -- deleting the `return Err`, not the condition that reaches it.
  "src/apps.rs|    let lambda = ((1.0 - flip_probability) / flip_probability).ln();|    let lambda = 1.0;|apps::tests::the_field_is_the_channels_log_odds_and_not_a_tunable_weight|a data term that forgets the channel's log-odds"
  "src/apps.rs|            b.bias(i, t * lambda / 2.0);|            b.bias(i, t * lambda);|apps::tests::an_mpm_estimate_beats_the_exact_map_on_pixel_error|a data term twice as strong as the channel says"
  "src/apps.rs|                b.couple(i, i + w, smoothness);|                b.couple(i, i + w, smoothness * 0.5);|apps::tests::the_field_is_the_channels_log_odds_and_not_a_tunable_weight|a prior that smooths vertically less than horizontally"
  "src/apps.rs|    let draws = (traces.len() * traces.first().map_or(0, Vec::len)) as f64;|    let draws = (traces.len() * traces.first().map_or(0, Vec::len) / 2) as f64;|apps::tests::the_sampled_posterior_converges_to_the_exact_one|a posterior normalised by half the draws it averaged"
  "src/apps.rs|        return Err(Refused::Smoothness(smoothness));|        return Ok(GraphBuilder::new(1).build());|apps::tests::an_ill_posed_task_is_refused_by_name|a supermodular smoothness accepted, breaking the MAP oracle"
  "src/apps.rs|        return Err(Refused::FlipProbability(flip_probability));|        return Ok(GraphBuilder::new(1).build());|apps::tests::an_ill_posed_task_is_refused_by_name|a flip probability of 0 or 0.5 accepted"
  "src/apps.rs|        && chains < 2|        && chains < 1|apps::tests::an_ill_posed_task_is_refused_by_name|a single chain graded on its own trace"

  # THE PATH TO EMITTED HARDWARE, and the diagnostic that nearly let it lie. `apps::restore`'s
  # fabric route builds the p-bit netlist this crate emits for an FPGA, so an image becomes a
  # posterior becomes a netlist. On that fabric a chain's randomness is a RESET CONSTANT of the
  # netlist, which makes "one seed for every chain" the likely mistake rather than a contrived one.
  #
  # ROW 3 SURVIVED ITS FIRST RUN, AND EXPOSED A LIBRARY DEFECT RATHER THAN A TEST GAP. Four
  # IDENTICAL chains passed `floors::Convergence::from_chains`: identical chains have zero
  # between-chain variance, so R-hat collapses to sqrt((N-1)/N) and reads 1.00000 against the
  # 1.00010 four genuinely independent chains give -- it scores the degenerate case BETTER. The
  # diagnostic whose whole purpose is to catch chains that are not exploring independently cannot
  # see chains that are not independent at all. `Convergence::Identical` now refuses them, and the
  # last row below is that guard. (`is_certified` then certified the new variant on the day it was
  # added, being `!matches!(self, SingleChain)` -- a negation grants every future variant the
  # affirmative answer. Now stated positively; run by hand and caught, but its text contains an
  # or-pattern pipe so it cannot be a row.)
  #
  # Run and found EQUIVALENT: changing the hardware gate's sweep count. Both the emulator's target
  # and the testbench's read the same constant, so the comparison stays consistent and the mutant
  # changes nothing. Recorded, not contrived around.
  "src/apps.rs|                if c > 0 {|                if false {|apps::tests::the_fabric_route_restores_and_bills_in_the_unit_that_was_metered|a netlist per chain that is not charged as a load"
  "src/apps.rs|                    cost.reads += n as u64;|                    cost.reads += 1;|apps::tests::the_fabric_route_restores_and_bills_in_the_unit_that_was_metered|a whole-state readback billed as a single read"
  "src/apps.rs|                    seed ^ (c as u64).wrapping_mul(0x9E37_79B9),|                    seed,|apps::tests::the_fabric_route_restores_and_bills_in_the_unit_that_was_metered|every chain given the same generator seed"
  "src/apps.rs|                    1.0, // beta is 1: the energy already IS the negative log posterior|                    2.0, // beta is 1: the energy already IS the negative log posterior|apps::tests::the_fabric_route_restores_and_bills_in_the_unit_that_was_metered|a fabric quantised at the wrong temperature"
  "src/floors.rs|                if a.len() == b.len() && !a.is_empty() && a == b {|                if false {|floors::tests::four_identical_chains_are_refused_by_name_because_rhat_cannot_see_them|the identical-chain check removed, leaving R-hat blind"

  # A CERTIFIED OPTIMALITY GAP ON A MULTI-LABEL MRF. Three labels is NP-hard, so the answer cannot
  # be checked against an optimum at any useful size -- it arrives with a lower bound instead, and
  # these rows are what stand between that bound and a number that merely looks like one.
  #
  # Two mutants were run and found EQUIVALENT ON THE LP FIXTURE and are filed against soundness
  # instead: dropping the reweighting leaves the triangle's tree minima unchanged (its rho is 2/3
  # and the minimiser does not move), so the fixture that pins D = 1 < LP = 1.5 < E_min = 2 cannot
  # see it, while the 24-instance soundness sweep can. A fixture that pins one number exactly is
  # not thereby sensitive to everything upstream of it.
  "src/mrf.rs|            w * forest_min_energy(m, &reweighted)|            forest_min_energy(m, &reweighted)|mrf::tests::the_fixed_splitting_is_weaker_than_the_lp_bound_and_says_so|forest minima summed without their convex weights"
  "src/mrf.rs|                    let scaled = c / cover.rho[e];|                    let scaled = c;|mrf::tests::neither_route_is_ever_above_the_true_minimum|the tree reweighting dropped, so the forests average to a weaker model"
  "src/mrf.rs|        let l = -u / beta;|        let l = -u * beta;|mrf::tests::neither_route_is_ever_above_the_true_minimum|a log partition turned into an energy by multiplying"
  "src/mrf.rs|                b.field(i, a as u8, -costs[i * q + a]);|                b.field(i, a as u8, costs[i * q + a]);|mrf::tests::a_structured_instance_certifies_far_tighter_than_an_adversarial_one|a data cost entered with the wrong sign"
  "src/mrf.rs|            message[i] = acc.clone();|            message[i] = vec![0.0f64; q];|mrf::tests::a_forests_log_partition_and_minimum_are_exact_against_enumeration|a min-sum root that drops its own field"
  "src/mrf.rs|                if c < best_c - 1e-15 {|                if c > best_c + 1e-15 {|mrf::tests::icm_descends_to_a_local_minimum_from_a_deliberately_bad_start|iterated conditional modes ascending instead of descending"
  "src/mrf.rs|    let gap = m.single_site_gap_max().unwrap_or(1.0);|    let gap = 1.0;|mrf::tests::rescaling_every_cost_does_not_change_the_labelling|an annealing ladder in absolute beta rather than the model's units"
  "src/mrf.rs|        bound: crate::round::sum_down(&terms),|        bound: crate::round::sum_up(&terms),|mrf::tests::the_weighted_sum_rounds_in_the_direction_that_keeps_each_bound_sound|a lower bound rounded upward, away from soundness"

  # GAUSSIAN BELIEF PROPAGATION, and the application entry point over it. Its two classical facts
  # are load-bearing here: exact on a tree in BOTH moments, and on a loop the mean stays exact
  # while the variance does not. The tree rows below break the first, which is the only way to
  # tell a correct implementation from one that is wrong everywhere -- a module whose variances
  # were merely wrong would still "reproduce" the loopy finding.
  "src/gbp.rs|                    let np = -w * w / cav_p;|                    let np = w * w / cav_p;|gbp::tests::on_a_tree_belief_propagation_is_exact_in_both_moments|a message precision with the wrong sign"
  "src/gbp.rs|                    let cav_p = tot_p - m_prec[back];|                    let cav_p = tot_p;|gbp::tests::on_a_tree_belief_propagation_is_exact_in_both_moments|a cavity that does not exclude the reply, the classic BP bug"
  "src/gbp.rs|                    let cav_e = tot_e - m_info[back];|                    let cav_e = tot_e;|gbp::tests::on_a_tree_belief_propagation_is_exact_in_both_moments|a cavity information that does not exclude the reply"
  "src/gbp.rs|            variance[i] = 1.0 / p;|            variance[i] = p;|gbp::tests::on_a_tree_belief_propagation_is_exact_in_both_moments|a variance reported as a precision"
  "src/gbp.rs|        if !(residual <= tol) {|        if false {|gbp::tests::a_model_that_does_not_settle_is_refused_rather_than_reported|an unsettled run returned as a belief"
  "src/gbp.rs|            if !seen.insert(key) {|            if seen.insert(key) {|gbp::tests::an_ill_formed_model_is_named_rather_than_solved|a duplicate-edge check inverted"

  # THE LAST ROW SURVIVED ITS FIRST RUN. The comparison asserted only that the sampler cost MORE
  # node updates than message passing, and billing per step instead of per node still cleared that
  # by 79x. Counts are asserted exactly now, as the image task's already were.
  "src/apps.rs|                samples: r.steps * n as u64,|                samples: r.steps,|apps::tests::three_routes_to_one_gaussians_marginals_and_only_two_have_the_variance|a sampling route billed per step, not per node update"
  "src/apps.rs|                reads: (samples as u64) * (n as u64),|                reads: (samples as u64),|apps::tests::three_routes_to_one_gaussians_marginals_and_only_two_have_the_variance|a readback billed per draw, not per node per draw"

  # BAYESIAN LOGISTIC REGRESSION -- a classifier's CONFIDENCE, where the sampler's case is again the
  # second moment. The oracle is grid quadrature, and the rows below break the model, the oracle
  # and the cheap route in turn.
  #
  # ROW 3 SURVIVED ITS FIRST RUN, AND THE REASON IS GENERAL. The oracle test checked that refining
  # the grid and widening the box stopped moving the answer -- which a WRONG INTEGRAND does just as
  # obediently. A convergence check is not a correctness check. The test now holds the quadrature
  # against an independent answer (with no data the posterior IS the prior, so the predictive is a
  # one-dimensional Gaussian-logistic integral from another routine) and against the fact that a
  # probability lies in [0, 1], which the logit-averaging mutant misses by reaching 7.75.
  #
  # Run by hand and caught but not rows: a prior twice as strong as declared (its text contains a
  # closure, so a pipe). Run and found EQUIVALENT on these fixtures: centring the quadrature box on
  # the origin rather than the mode -- every box in the test is wide enough that its centre cannot
  # change the answer, which is exactly what the widening check establishes. Recorded, not contrived
  # around.
  "src/logit.rs|            let c = -s * sigmoid(-s * self.row_dot(i, q));|            let c = s * sigmoid(-s * self.row_dot(i, q));|logit::tests::the_potential_and_its_gradient_are_what_the_model_says_they_are|a likelihood gradient with the wrong sign"
  "src/logit.rs|            u += softplus(-s * self.row_dot(i, w));|            u += softplus(s * self.row_dot(i, w));|logit::tests::the_potential_and_its_gradient_are_what_the_model_says_they_are|a potential that swaps the labels"
  "src/logit.rs|                            num[k] += p * sigmoid(a_q);|                            num[k] += p * a_q;|logit::tests::the_quadrature_oracle_has_stopped_moving_before_anything_is_measured_against_it|a quadrature averaging the logit, not the probability"
  "src/logit.rs|                        den += p;|                        den += 1.0;|logit::tests::the_quadrature_oracle_has_stopped_moving_before_anything_is_measured_against_it|a quadrature that forgets to weight its denominator"
  "src/logit.rs|        let hw = half_width * self.prior_sd;|        let hw = half_width;|logit::tests::the_quadrature_oracle_has_stopped_moving_before_anything_is_measured_against_it|a quadrature box whose width ignores the prior's scale"
  "src/logit.rs|                if d != 2 {|                if false {|logit::tests::the_quadrature_oracle_has_stopped_moving_before_anything_is_measured_against_it|grid quadrature offered above two dimensions"
  "src/logit.rs|                        gauss_logistic(mu, var.max(0.0).sqrt())|                        sigmoid(mu)|logit::tests::the_cheap_routes_get_the_decision_right_and_bracket_the_confidence_without_bounding_it|a Laplace route collapsed to the plug-in"

  # The frame data register is READ during a capture and WRITTEN during configuration, and the two
  # headers differ by one field. Issue the write form and the device loads the frames it was meant
  # to fetch, which is a readback that silently reconfigures the part.
  "silicon/src/capture.rs|        type1_read(reg::FDRO, 0),|        type1_write(reg::FDRO, 0),|capture::tests::the_stream_captures_before_it_reads|a frame readback issued as a frame write|ferrotherm-silicon"

  # Frame addresses increment by one inside a column and stop doing so at its end, where the 7-series
  # minor field wraps at 128. Let a run past the boundary through and every frame after it carries an
  # address it did not come from.
  "silicon/src/capture.rs|        if minor as usize + n_frames > FRAMES_PER_COLUMN {|        if minor as usize + n_frames > 256 {|capture::tests::a_run_past_the_column_end_is_refused|a readback run that walks off its column|ferrotherm-silicon"

  # A sample that reached none of the located cells has no disagreement in it, which reads exactly
  # like a clean result. Same shape as a cross-check that verifies because it compared nothing.
  "silicon/src/capture.rs|        !self.values.is_empty()|        self.values.len() < usize::MAX|capture::tests::a_sample_that_covered_nothing_is_not_an_observation|a readout that observes nothing and says so|ferrotherm-silicon"

  # A located cell is anonymous without its latch name: a captured frame is 3,232 bits and the name
  # is the only thing that says which one is the flip-flop. Trim the prefix wrong and every name
  # keeps a leading `=`, which prints plausibly and matches nothing.
  "silicon/src/logic_location.rs|            if let Some(name) = field.strip_prefix(\"Latch=\") {|            if let Some(name) = field.strip_prefix(\"Latch\") {|logic_location::tests::parses_the_documented_line_shape_and_skips_the_rest|a latch name that keeps its separator|ferrotherm-silicon"

  # The analogue floor is 12 kT 2^{2b}, and the exponent is the whole argument: it is what makes the
  # ratio against Landauer grow like 2^{2b}/b instead of staying near one. Halve the exponent and
  # every number in the table is still a plausible energy and the conclusion is gone.
  "src/precision.rs|    kappa * BOLTZMANN_CONSTANT * temperature_k * 2.0f64.powi(2 * bits as i32)|    kappa * BOLTZMANN_CONSTANT * temperature_k * 2.0f64.powi(bits as i32)|precision::tests::the_thermal_bound_is_independent_of_voltage_and_quadruples_per_bit|an analogue floor that doubles per bit instead of quadrupling"

  # `divergence` and `solenoidal_divergence` are deliberately separate functions because conflating
  # them is exactly how an asymmetric coupling gets mistaken for a measure-preserving perturbation.
  # Take the full coupling here and the solenoidal part stops being identically zero, which turns
  # the finding into its opposite while every number stays finite and plausible.
  "src/kuramoto.rs|                terms.push(-a[i * n + j] * (phi[j] - phi[i]).cos());|                terms.push(-self.k[i * n + j] * (phi[j] - phi[i]).cos());|kuramoto::tests::an_antisymmetric_coupling_is_divergence_free_but_not_gibbs_preserving|a solenoidal divergence taken over the full coupling"

  # Lifting is exactly the reversal on rejection. Drop it and the chain proposes one direction
  # forever: it still runs, still accepts by Metropolis, still looks like a sampler, and no longer
  # has the target as its stationary law. Only the sweep-level test sees this -- the matrix-level
  # stationarity tests are about `lifted_matrix` and cannot fail for it.
  "src/nonrev.rs|                self.v[i] = -step;|                self.v[i] = step;|nonrev::tests::a_lifted_sweep_reproduces_the_enumerated_distribution|a lifted chain that never reverses"

  # The whole point of lowering their instruction set is that connectivity is a WRITE, and a write is
  # the expensive operation. Price it as a read and the configuration share collapses, which is the
  # answer the architecture would like.
  "src/dsisa.rs|        let config = self.configuration_writes as f64 * prices.e_write;|        let config = self.configuration_writes as f64 * prices.e_read;|dsisa::tests::configuration_can_cost_more_than_computation|a configuration phase priced as reads"

  # The von Mises conditional is (R, phi) with phi = atan2(sin, cos). Transpose the arguments and the
  # concentration is still right, the sampler still draws a circular distribution, and the mean angle
  # is reflected about pi/4 -- a generator that samples smoothly from the wrong law.
  "src/phasegen.rs|        (c.hypot(s), s.atan2(c))|        (c.hypot(s), c.atan2(s))|phasegen::tests::the_phase_conditional_is_exactly_the_von_mises_the_field_names|a von Mises mean angle with its arguments transposed"

  # The noise a Langevin step injects IS the temperature. Halve its power and the chain still runs,
  # still looks stationary, and samples a distribution twice as cold -- which is the failure mode an
  # identity proof cannot see, because the identities are about the drift.
  "src/nonrev.rs|        let sigma = (2.0 * self.dt / self.beta).sqrt();|        let sigma = (1.0 * self.dt / self.beta).sqrt();|nonrev::tests::the_skew_drift_leaves_a_gaussian_target_where_it_found_it|a Langevin step at half the temperature it states"

  # A grid partition function is the model's plus one cell volume per phase, and it rises by n ln 2
  # for every doubling of q. Return it unadjusted and the number quoted for the model is a property
  # of the readout instead -- 7.33 at 16 points, 8.72 at 32, for the same model.
  "src/phasegen.rs|        self.log_z - self.n as f64 * (self.q as f64 / TAU).ln()|        self.log_z|phasegen::tests::the_partition_function_is_the_models_not_the_grids|a partition function that moves with the grid it was read on"

  # A degree budget is met only if BOTH endpoints agree to keep a coupling. Take each node's heaviest
  # and union them and a popular node ends up above the budget -- a graph the fabric still cannot
  # wire, reported as one that fits.
  "src/kuramoto.rs|                if top[i * n + j] && top[j * n + i] {|                if top[i * n + j] {|kuramoto::tests::keeping_each_nodes_heaviest_is_not_enough_to_bound_the_degree|a degree budget met by only one endpoint"

  # The dropped mass IS the bound. Fail to count a coupling that was dropped and the bound reports
  # zero cost for a truncation that moved the distribution, which is a bound in the unsafe direction.
  "src/kuramoto.rs|                    dropped_terms.push(w.abs());|                    dropped_terms.push(0.0);|kuramoto::tests::the_truncation_bound_holds_against_the_measured_kl|a dropped coupling that is not counted"

  # The factor of two is what makes the truncation gap and the covering gap the same currency: both
  # are 2 beta times a mass in energy units. Halve one and they stop adding.
  "src/kuramoto.rs|        2.0 * self.dropped_mass * beta|        1.0 * self.dropped_mass * beta|kuramoto::tests::the_covering_and_truncation_gaps_are_one_currency|a truncation bound at half strength"

  # The parameter gradient belongs to the oscillator the cotangent is ON, not the one the coupling
  # points at. Swap the endpoint and every entry of dL/dK is the transpose of itself -- a gradient
  # that still descends on a symmetric model and trains the wrong thing on an asymmetric one, which
  # is the only kind the published architecture has.
  "src/pathwise.rs|                d_k[i * n + j] += dt * lam[i] * (phi[j] - phi[i]).sin();|                d_k[i * n + j] += dt * lam[j] * (phi[j] - phi[i]).sin();|pathwise::tests::the_adjoint_matches_central_differences|an adjoint indexed by the wrong endpoint"

  # The reverse pass applies the TRANSPOSE of the Jacobian. Apply the Jacobian itself and the
  # gradient is finite, smooth, and of a different function.
  "src/pathwise.rs|                terms.push(jac[i * n + j] * lam[i]);|                terms.push(jac[j * n + i] * lam[i]);|pathwise::tests::the_adjoint_matches_central_differences|a Jacobian transposed the wrong way in the reverse pass"

  # The reference integrator is the standin for the differential equation, so its order IS the
  # argument. Drop one of the middle weights and it silently becomes second order -- still far
  # better than Euler, still converging, and no longer a reference.
  "src/pathwise.rs|        phi[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);|        phi[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + k3[i] + k4[i]);|pathwise::tests::the_reference_integrator_is_fourth_order|a Runge-Kutta weight that drops the order"

  # Mitchell's worst multiply is exactly 1/9, at three halves, and the constant is the whole claim a
  # logarithmic datapath has to answer. State it as anything else and every correction measured
  # against it is measured against the wrong bar.
  "src/logdomain.rs|pub const MAX_MUL_RELATIVE_ERROR: f64 = 1.0 / 9.0;|pub const MAX_MUL_RELATIVE_ERROR: f64 = 1.0 / 8.0;|logdomain::tests::mitchells_worst_multiply_is_exactly_one_ninth|a worst case stated as one eighth"

  # The forward and inverse corrections are DIFFERENT functions. Leave the reconstruction raw -- the
  # obvious first implementation -- and the error plateaus near a percent no matter how much table
  # is thrown at it, while every table-size number still falls and looks like progress.
  "src/logdomain.rs|            inverse.push(2.0f64.powf(mid) - (1.0 + mid));|            inverse.push(0.0);|logdomain::tests::correcting_mitchell_to_fp16_accuracy_costs_ten_bits_of_table|a correction that leaves the reconstruction raw"

  # Amdahl's denominator is the share you did NOT touch plus the share you did, divided. Drop the
  # first term and a datapath optimisation reports the speed-up of the datapath as the speed-up of
  # the device, which is the arithmetic every accelerator claim turns on.
  "src/logdomain.rs|    let terms = [1.0 - share, share / factor];|    let terms = [1.0, share / factor];|logdomain::tests::a_free_multiplier_is_still_bounded_by_what_it_was_a_share_of|an Amdahl denominator that forgets the untouched share"

  # A ratio between two energy figures is only as good as its WEAKER side. Return the stronger and a
  # metered number divided by a pre-silicon projection reports as metered -- which is the single most
  # common error in the literature this type was written against.
  "src/ledger.rs|    if a <= b { a } else { b }|    if a >= b { a } else { b }|ledger::tests::a_comparison_is_only_as_good_as_its_weaker_side|a comparison graded by its stronger side"

  # The TUR floor scales with the SQUARE of the signal-to-noise ratio, and the square is the whole
  # finding: it is what makes a coarse answer cheap and a sharp one expensive in exact proportion.
  # Linear in SNR and the floor still rises with precision, still looks like a bound, and prices
  # every sampling machine wrong by a factor of the SNR.
  "src/floors.rs|    2.0 * BOLTZMANN_CONSTANT * temperature_k * snr * snr|    2.0 * BOLTZMANN_CONSTANT * temperature_k * snr|floors::tests::you_pay_for_the_precision_you_extract|a sampling floor linear in SNR rather than squared"

  # Energy, time and ACCURACY. Drop the quality term and the bound says dissipation falls to zero as
  # the drive slows, which is the exact misreading the published inequality exists to prevent.
  "src/floors.rs|    wasserstein_sq / (beta * gamma * tau_drive * (1.0 - quality))|    wasserstein_sq / (beta * gamma * tau_drive)|floors::tests::the_energy_time_accuracy_bound_trades_three_ways|an energy-time bound that forgets accuracy"

  # The Bures-Wasserstein distance between Gaussians is built from the square ROOTS of the
  # variances. Use the variances and the distance is finite, positive, zero when the laws agree --
  # and wrong, so every floor computed from it is wrong in an unsafe direction.
  "src/floors.rs|        let r = s0[i].sqrt() - s1[i].sqrt();|        let r = s0[i] - s1[i];|floors::tests::the_wasserstein_distance_matches_its_closed_form|a Bures distance on variances rather than their roots"

  # A per-sample cost divided by SWEEPS rather than by independent samples flatters a correlated
  # chain by exactly its autocorrelation time -- which for a cold fabric is three orders of
  # magnitude, and is the number this crate measures everywhere else.
  "src/floors.rs|    let effective_samples = sweeps / (2.0 * tau_int);|    let effective_samples = sweeps;|floors::tests::a_metered_run_can_be_priced_against_its_own_floor|a per-sample cost that ignores autocorrelation"

  # ---- exact verification at scale ---------------------------------------------------------------
  # An interval so wide it cannot miss reports agreement with every hypothesis at once, including
  # wrong ones. Widen the 95% band tenfold: the ladder test requires the interval at n = 144 to
  # EXCLUDE a lattice one percent colder, and almost nothing else in the suite asks `covers` to
  # say no. Rows above prove a test notices a broken value; this one proves a test notices a
  # meaningless error bar.
  "src/samples.rs|self.value - 1.96 * self.stderr, self.value + 1.96 * self.stderr|self.value - 19.6 * self.stderr, self.value + 19.6 * self.stderr|pfaffian::tests::an_exact_check_discriminates_better_as_the_lattice_grows|a confidence interval ten times too wide to miss"

  # ---- the convergence term the pricing did not have -------------------------------------------
  # Both of these shipped: `cost_per_effective_sample` divided sweeps by `2 tau_int` and took no
  # convergence input at all. A chain stuck in its starting basin has a SMALL tau_int -- it only
  # fluctuates inside that basin -- so it priced as many cheap independent samples. The error runs
  # in the direction that flatters the run, on the one number this crate exists to report honestly.
  # Disable either refusal and the named test must notice.
  "src/floors.rs|} else if burn_in < coalesced_at {|} else if false {|floors::tests::a_run_that_did_not_converge_is_refused_rather_than_priced|a burn-in shorter than coalescence required, priced anyway"

  "src/floors.rs|} else if rhat > Convergence::RHAT_LIMIT {|} else if false {|floors::tests::a_run_that_did_not_converge_is_refused_rather_than_priced|chains that do not agree, priced anyway"

  # The first `refusal` rejected any R-hat under one as "not real". A sample R-hat is
  # sqrt((N-1)/N + B/(N W)), so well-mixed chains land just UNDER one -- four real chains gave
  # 0.99985 -- and the rule refused exactly the best-converged runs. The test beside it checked 1.0
  # and 0.5 and never a realistic value. Put the old floor back and real chains must notice.
  "src/floors.rs|} else if rhat < Convergence::RHAT_FLOOR {|} else if rhat < 1.0 {|floors::tests::convergence_from_chains_takes_the_strictest_diagnostic_and_refuses_too_little_evidence|an R-hat just under one refused as impossible"

  # `from_chains` takes the LARGEST of three R-hats because each is blind where another sees.
  # Collapse them to the plain one and chains with equal means and different spreads -- which
  # plain R-hat passes, by an assertion in the same test -- would be priced.
  "src/floors.rs|let all = [d.rhat, d.rhat_rank, d.rhat_folded];|let all = [d.rhat, d.rhat, d.rhat];|floors::tests::convergence_from_chains_takes_the_strictest_diagnostic_and_refuses_too_little_evidence|the strictest diagnostic replaced by the most forgiving"

  # ---- the bounding chain ---------------------------------------------------------------------
  # An unknown neighbour must widen the field bracket by its FULL coupling, both ways. Narrow it
  # and the chain still "coalesces" -- faster, in fact -- onto a draw that is not from the model,
  # with nothing in the output to say so. On a frustrated model the exactness test must notice.
  "src/cftp.rs|                    let a = g.w[k].abs();|                    let a = 0.0 * g.w[k].abs();|cftp::tests::a_frustrated_model_is_sampled_exactly_by_the_bounding_chain|an unknown neighbour that widens nothing"

  # ---- the total-correlation penalty's SIGN ------------------------------------------------------
  # `examples/dtm_scale` carried this term reversed for as long as it existed: it ASCENDED total
  # correlation, driving the model to be more correlated than the data, and because the reversed
  # update also has a fixed point, watching |J| settle never noticed. The test decides the direction
  # by the conditional's exact total correlation. Reverse the library's sign and it must go red.
  "src/dtm.rs|                g += lambda_tc * -(ma * mb - neg_ss[k] / m);|                g += lambda_tc * (ma * mb - neg_ss[k] / m);|dtm::tests::the_total_correlation_term_descends_the_conditionals_exact_total_correlation|a correlation penalty that rewards correlation"

)

bad=0
ran=0
unevaluated=0
skipped_rows=""
# EVERY FILTER MUST NAME EXACTLY ONE TEST, and until an audit asked, fourteen did not: two rows
# filtered on `bound`, which matches 54 of the 906 lib tests. A broad filter still goes red, so the
# suite still reports "caught" -- but it is then evidence that SOMETHING noticed, not that the test
# the row names did. Two of the fourteen named a test that could not have caught the mutation at
# all. This resolves each filter against the real test list before any mutation is applied.
check_filters() {
  local list_root list_pkg row file old new filter label pkg n
  list_root=$(cargo test --release --lib -- --list 2>/dev/null | sed -n 's/: test$//p')
  local bad=0
  for row in "${mutations[@]}"; do
    IFS='|' read -r file old new filter label pkg <<<"$row"
    if [ -n "${pkg:-}" ]; then
      list_pkg=$(cargo test --release -p "$pkg" -- --list 2>/dev/null | sed -n 's/: test$//p')
    else
      list_pkg="$list_root"
    fi
    n=$(printf '%s\n' "$list_pkg" | grep -cF -- "$filter" || true)
    if [ "$n" -ne 1 ]; then
      echo "filter '$filter' names $n tests, not 1 -- '$label'" >&2
      bad=$((bad + 1))
    fi
  done
  if [ "$bad" -gt 0 ]; then
    echo "$bad row(s) do not name exactly one test. A row is evidence about the test it names." >&2
    exit 2
  fi
}
check_filters

# EVERY OLD STRING MUST STILL BE IN ITS FILE, EXACTLY ONCE. A refactor can remove the line a row
# mutates and leave the row measuring nothing; the 127-row run of 2026-09-13 found row 118 that way,
# two hours in ("MUTATION DID NOT APPLY"), because single-row checks only ever exercise new rows.
# This resolves every row's target against the source before any mutation is applied.
check_olds() {
  local row file old new filter label pkg n bad=0
  for row in "${mutations[@]}"; do
    IFS='|' read -r file old new filter label pkg <<<"$row"
    if [ ! -f "$file" ]; then
      echo "file '$file' does not exist -- '$label'" >&2
      bad=$((bad + 1))
      continue
    fi
    n=$(grep -cF -- "$old" "$file" || true)
    if [ "$n" -ne 1 ]; then
      echo "old string occurs $n times in $file, not 1 -- '$label'" >&2
      bad=$((bad + 1))
    fi
  done
  if [ "$bad" -gt 0 ]; then
    echo "$bad row(s) no longer name a line that exists once. A stale row is a row that measures nothing." >&2
    exit 2
  fi
}
check_olds

# FERROTHERM_MUTATION_PRECHECK_ONLY=1 stops here: the filters and targets have been resolved and
# nothing has been mutated. A two-minute preflight for a two-hour run.
if [[ "${FERROTHERM_MUTATION_PRECHECK_ONLY:-0}" = "1" ]]; then
  echo "precheck only: ${#mutations[@]} rows, every filter names one test and every target exists once"
  exit 0
fi

# THE COUNT IS PINNED. A row deleted in a merge, or commented out to get a build green, leaves a
# suite that still says "all mutations caught" over a smaller set -- which reads exactly like
# success. Nothing anywhere asserted how many rows there should be until an audit asked.
expected_rows=272
if [ "${#mutations[@]}" -ne "$expected_rows" ]; then
  echo "the suite has ${#mutations[@]} rows and expects $expected_rows." >&2
  echo "adding rows is good -- raise expected_rows. Losing one silently is what this catches." >&2
  exit 2
fi

for row in "${mutations[@]}"; do
  IFS='|' read -r file old new filter label pkg <<<"$row"
  # `|` IS THE SEPARATOR, so a mutation whose code contains one shifts every field after it: the
  # replacement becomes half a pattern, the test filter becomes a fragment of code, and the row
  # reports a pass over nothing. Rejoining and comparing is the whole check, and it is here because
  # a row added for `exact_gradient` did exactly this -- `vkey | (c << data.visible)` is ordinary
  # Rust and an unreadable table row.
  rejoined="$file|$old|$new|$filter|$label${pkg:+|$pkg}"
  if [ "$rejoined" != "$row" ]; then
    echo "malformed row: a field contains the '|' separator, so the fields below are shifted." >&2
    echo "  row:  $row" >&2
    echo "  read: $rejoined" >&2
    bad=$((bad + 1))
    continue
  fi
  out="$(scripts/mutation-check.sh "$file" "$old" "$new" "$filter" "$label" ${pkg:+"$pkg"} 2>&1)"
  echo "$out"
  ran=$((ran + 1))
  # RED is the only good outcome. Every other line means this row is not evidence of anything:
  # a mutation that did not apply tests nothing, a filter matching no test ran nothing, and a
  # build failure is inconclusive rather than safe.
  #
  # One case is about the MACHINE rather than the row. A test gated on hardware this runner does
  # not have — no GPU adapter, no power sensor — skips, and a skipping test PASSES, which is
  # indistinguishable from a surviving mutant unless it is asked. CI called an absent GPU a blind
  # test for exactly this reason. Counted apart, and not silently forgiven:
  # `FERROTHERM_REQUIRE_ALL=1` turns it fatal, the same lever `check-semantics` and `check-answers`
  # already use for this situation.
  if grep -q "NOT EVALUATED HERE" <<<"$out"; then
    unevaluated=$((unevaluated + 1))
    skipped_rows="$skipped_rows
  - $label"
  elif ! grep -q "RED (good)" <<<"$out"; then
    bad=$((bad + 1))
  fi
done

echo
if [[ $ran -eq 0 ]]; then
  echo "no mutations defined, so this proved nothing" >&2
  exit 2
fi
if [[ $bad -gt 0 ]]; then
  echo "$bad of $ran mutations were not caught by the test named for them." >&2
  echo "A suite that stays green while the code is wrong is not measuring the code." >&2
  exit 1
fi
if [[ $unevaluated -gt 0 ]]; then
  echo "$unevaluated of $ran mutations were NOT EVALUATED on this machine:$skipped_rows" >&2
  echo >&2
  echo "Their tests need hardware this runner does not have, so they skipped — and a skipping" >&2
  echo "test passes, which is exactly what a surviving mutant looks like. This run is not" >&2
  echo "evidence about those rows in either direction. Run it where the hardware is, or set" >&2
  echo "FERROTHERM_REQUIRE_ALL=1 to make it fatal." >&2
  if [[ "${FERROTHERM_REQUIRE_ALL:-0}" = "1" ]]; then
    exit 1
  fi
fi
if [[ $unevaluated -gt 0 ]]; then
  echo "  $((ran - unevaluated)) of $ran mutations caught; $unevaluated not evaluated here"
else
  echo "  all $ran mutations caught by the test named for each"
fi
