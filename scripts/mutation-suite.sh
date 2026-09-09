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
  "src/lp.rs|[name, \">=\", _] => err(|[name, \">=\", lo] => return Ok((num(lo)?, name.to_string(), num(lo)? + 1)), #[allow(unreachable_patterns)] [name, \">=\", _] => err(|lp::|an unbounded LP bound"

  # Determinism. HashMap iteration order decides CSR neighbour order, which decides the order every
  # local field is summed in, which changes the last bits of the energy. Five runs, five programs.
  "src/graph.rs|BTreeMap<(u32, u32), f64> = std::collections::BTreeMap::new()|HashMap<(u32, u32), f64> = std::collections::HashMap::new()|graph::|CSR order via HashMap"

  # A NaN objective coefficient compiled, solved, and reported feasible while silently disabling
  # every other preference -- comparisons against NaN are all false, so the sampler stopped
  # improving. Remove the guard and the refusal test must notice.
  "src/model.rs|self.objective.check_finite(\"objective coefficient\")?;|let _ = self.objective.check_finite(\"objective coefficient\");|not_finite|NaN objective coefficient"

  # scale_to_fit bounded the ANSWER instead of the work: above a 1e6 candidate ceiling it refused
  # without trying the largest candidate, which is usually the answer.
  # Reintroduced through the loop guard rather than the `if`, because the original defect was
  # written `|| top > 1e6` and a literal `|` is this table's field separator.
  "src/fabric.rs|while n >= 1.0 && tried < 1_000_000 {|while n >= 1.0 && tried < 1_000_000 && top <= 1e6 {|wide_integral|scale_to_fit ceiling refusal"

  # `Sum for f64` folds from -0.0, so a model with no soft violations reported \"-0\".
  "src/model.rs|.sum::<f64>() + 0.0|.sum::<f64>()|soft|soft cost negative zero"

  # A `Device::run` that accepts a seed and swallows it reports reproducibility it does not have:
  # the caller varies the seed, gets one answer every time, and reads a deaf sampler as a confident
  # one. `conform`'s determinism case CANNOT catch this -- an ignored seed is perfectly
  # reproducible -- so the tests that can are the ones named here.
  "gpu/src/lib.rs|let offset = (seed as u32).wrapping_mul(0x9E37_79B9);|let offset = 0u32; let _ = seed;|seed|Device::run swallows its seed|ferrotherm-gpu"

  # A BUSY MACHINE HAS NO IDLE. `Meter::idle` guarded the DELTA against the baseline's noise and
  # never checked the baseline was idle, so a complete energy table was published here from a
  # machine at load 82 -- other sessions' work charged to this workload. The contamination inflates
  # the baseline, which overstates the idle share, which is the direction that flattered the very
  # argument being made. Leave the guard in place but make it always permit, and the test that
  # exercises the DECISION (rather than whatever this machine happens to be doing) must go red.
  "meter/src/lib.rs|Some(l) if l > QUIET_LOAD => Err(format!(|Some(l) if false && l > QUIET_LOAD => Err(format!(|quiet|idle baseline on a busy machine|ferrotherm-meter"

  # SOUNDNESS of the optimality bound. Drop one edge from the forest partition and `E = sum of the
  # parts` stops holding, so the "bound" describes a different problem -- still a number, still
  # plausible, and free to sit ABOVE the true minimum, which reports a NEGATIVE gap and makes every
  # conclusion drawn from it backwards. Checked against brute force on 200 random instances.
  "src/bound.rs|                forest.push((i, j, w));|                if !(i == 0 && j == 1) { forest.push((i, j, w)); }|bound|optimality bound drops an edge"

  # Subgradient ascent is not monotone -- 145 of 200 random instances peak before the last round --
  # so the bound must be the best round seen, not the final one. `if true` takes the last. This
  # mutation SURVIVED the whole suite on its first outing, because `forest(g, r)` already maximises
  # over rounds 0..r and no test could see inside that; `Bound::best_round` exists to make the
  # difference observable from outside.
  "src/bound.rs|        if total > best {|        if true {|bound|optimality bound takes the last round"

  # ---- the 0.44.0 receipt, whose defects all shipped once ----------------------------------------

  # THE READBACK, CHARGED WHERE IT HAPPENS. `anneal_scheduled` scores the whole state after every
  # sweep to keep a running best; that is a read of the whole state. It was charged by the CALLER,
  # so `Cpu::run` billed it and `Compiled::solve_with` -- the function a caller actually reaches
  # for -- billed nothing, pricing the identical anneal 244x apart. Worse, `KV260_MEASURED` states
  # no `e_read` precisely so a run that read cannot be priced against it, and the model API
  # returned a measured-silicon figure anyway. Set the charge to zero and the receipt test must go
  # red on BOTH the count and the refusal.
  "src/tempering.rs|                l.reads += g.n as u64;|                l.reads += 0;|receipt_tests|the readback goes uncharged"

  # A BEST-OF SEARCH PAYS FOR EVERY TRY. `best_of_all` sums the ledgers of all N; carrying only the
  # winner's makes an N-restart search read as one run. This shipped across the whole C ABI -- 1, 4
  # and 12 tries all reported 19,200 node updates -- because selection and aggregation lived in two
  # functions and only one of them was fixed.
  "src/model.rs|        winner.cost = total;|        let _ = total;|best_of_n|best-of carries only the winner"

  # THE FABRIC MUST DECLARE THE QUANTISATION IT PERFORMS. `Precision::Fixed` is scale-relative in
  # this crate (step = max|w| / levels); `FixedFabric` uses an absolute Q.8 grid. Under the wrong
  # declaration `check` computed ~0 relative error for a program whose weights were all 0.001 and
  # accepted it, and the fabric quantised every coupling to zero and sampled an empty graph.
  "src/hdl.rs|        f.coupling_precision = Precision::Grid { step: 1.0 / (1u32 << FRAC) as f64 };|        f.coupling_precision = Precision::Fixed { bits: 12 };|declared_precision|fabric mis-declares its grid"

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
  "src/informed.rs|            Balance::Sqrt => 0.5 * log_r,|            Balance::Sqrt => log_r,|informed|a balancing function that is not balanced"

  # Accepting everything is a valid-looking sampler with no Metropolis correction at all. It mixes
  # FASTER, which is the trap: a speed measurement would reward it.
  "src/informed.rs|            if after <= before || self.rng.f64() < before / after {|            if true {|informed|an informed chain that accepts everything"

  # The proposal is normalised by a total maintained incrementally. Flip the sign of the field
  # repair and the weights describe a state the sampler is not in.
  "src/informed.rs|            self.fields[j] += self.g.w[e] * 2.0 * sk;|            self.fields[j] -= self.g.w[e] * 2.0 * sk;|informed|an incremental field update with the wrong sign"

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
)

bad=0
ran=0
unevaluated=0
skipped_rows=""
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
