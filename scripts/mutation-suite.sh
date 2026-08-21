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

  # A BUSY MACHINE HAS NO IDLE. `Meter::idle` guarded the DELTA against the baseline's noise and
  # never checked the baseline was idle, so a complete energy table was published here from a
  # machine at load 82 -- other sessions' work charged to this workload. The contamination inflates
  # the baseline, which overstates the idle share, which is the direction that flattered the very
  # argument being made. Leave the guard in place but make it always permit, and the test that
  # exercises the DECISION (rather than whatever this machine happens to be doing) must go red.
  # A `Device::run` that accepts a seed and swallows it reports reproducibility it does not have:
  # the caller varies the seed, gets one answer every time, and reads a deaf sampler as a confident
  # one. `conform`'s determinism case CANNOT catch this -- an ignored seed is perfectly
  # reproducible -- so the tests that can are the ones named here.
  "gpu/src/lib.rs|let offset = (seed as u32).wrapping_mul(0x9E37_79B9);|let offset = 0u32; let _ = seed;|seed|Device::run swallows its seed|ferrotherm-gpu"

  "meter/src/lib.rs|Some(l) if l > QUIET_LOAD => Err(format!(|Some(l) if false && l > QUIET_LOAD => Err(format!(|quiet|idle baseline on a busy machine|ferrotherm-meter"
)

bad=0
ran=0
for row in "${mutations[@]}"; do
  IFS='|' read -r file old new filter label pkg <<<"$row"
  out="$(scripts/mutation-check.sh "$file" "$old" "$new" "$filter" "$label" ${pkg:+"$pkg"} 2>&1)"
  echo "$out"
  ran=$((ran + 1))
  # RED is the only good outcome. Every other line means this row is not evidence of anything:
  # a mutation that did not apply tests nothing, a filter matching no test ran nothing, and a
  # build failure is inconclusive rather than safe.
  grep -q "RED (good)" <<<"$out" || bad=$((bad + 1))
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
echo "  all $ran mutations caught by the test named for each"
