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
)

bad=0
ran=0
unevaluated=0
skipped_rows=""
for row in "${mutations[@]}"; do
  IFS='|' read -r file old new filter label pkg <<<"$row"
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
