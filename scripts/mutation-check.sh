#!/usr/bin/env bash
#
# Break the code on purpose, and require a named test to notice.
#
#   scripts/mutation-check.sh <file> <old> <new> <test-filter> <label>
#
# A green test suite says the code runs. It does not say the tests would catch the code being
# wrong, and the difference has been load-bearing in this project: tests have passed while the
# thing they described was broken, because the case they chose made the right and wrong answers
# coincide. This is how to find that out before shipping.
#
# THREE WAYS THIS HAS LIED HERE, each now checked for:
#
#   1. The mutation silently not applying. `sed`/`replace` patterns drift with the code, and a
#      pattern that matches nothing leaves the source correct — reported as "still green", which
#      reads exactly like a blind test. The helper below exits rather than pretending.
#   2. A mutation that breaks the BUILD. It produces neither "FAILED" nor "panicked", so grepping
#      for those reports a compile error as a passing test.
#   3. Grepping `^error` to catch (2) — which matches `error: test failed`, turning a genuine RED
#      into "inconclusive". The verdict comes from the `test result:` line and nothing else.
#
# A filter that matches no test is also called out: a mutation "surviving" a test that never ran
# is the emptiest result of all.
#
# RESTORES WITH GIT, so COMMIT FIRST. Twice in one day an uncommitted afternoon was destroyed this
# way — once by hand, once by this script's own cleanup.

set -uo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 <file> <old> <new> <test-filter> <label> [package]" >&2
  exit 2
fi
file="$1"; old="$2"; new="$3"; filter="$4"; label="$5"; pkg="${6:-}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

if ! git diff --quiet -- "$file"; then
  echo "refusing to run: $file has uncommitted changes, and this restores from git." >&2
  echo "commit first — that is the whole safety property." >&2
  exit 2
fi

# RESTORE ON ANY EXIT, NOT JUST THE HAPPY ONE. `git checkout -- "$file"` used to live only after
# the test run, so a kill between applying the mutation and reaching it left the MUTATED FILE in
# the working tree -- and a mutated file looks exactly like ordinary uncommitted work. It happened:
# a suite run was killed when its session ended, and `src/model.rs` sat there carrying
# `if false {` in place of `if !*hard {` until the next `git status` was read carefully. The next
# `git add -A` would have committed a deliberate defect with a green suite behind it.
#
# The trap fires on ordinary exit as well as INT, TERM and HUP, so the restore happens even if the
# test run is interrupted. It is idempotent: restoring an already-restored file is a no-op.
restore() { git checkout -- "$file" 2>/dev/null || true; }
trap restore EXIT INT TERM HUP

python3 - "$file" "$old" "$new" <<'PY' || { printf '  %-38s MUTATION DID NOT APPLY\n' "$label"; exit 0; }
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
t = open(path).read()
if old not in t:
    sys.exit(1)
open(path, 'w').write(t.replace(old, new, 1))
PY

# WHICH CRATE. `cargo test --lib` with no package tests the ROOT package only, so a mutation in
# `meter/`, `gpu/` or `serve/` would compile nothing and report NO TEST MATCHED -- a row that reads
# like a skip while testing nothing at all. Pass the package for those.
# `--nocapture`, so a test that SKIPPED can say so.
#
# A skipping test passes, and a passing test is exactly what a surviving mutant looks like. On a
# runner with no GPU adapter the `Device::run swallows its seed` row reported "STILL GREEN -- the
# test is blind" when the truth was that the test never ran: the three assertions that kill that
# mutant sit behind `dev_or_skip!`. Those are different facts and the suite could not tell them
# apart, so it called an absent GPU a blind test. Without `--nocapture` cargo swallows the skip
# message and the distinction is simply unavailable.
if [[ -n "$pkg" ]]; then
  out="$(cargo test --release --lib -p "$pkg" "$filter" -- --nocapture 2>&1)"
else
  out="$(cargo test --release --lib "$filter" -- --nocapture 2>&1)"
fi
# Restore here as well as in the trap: the verdict below is printed with the tree already clean, so
# anyone reading the output alongside `git status` sees what the next command would see.
restore

if ! grep -q "^test result:" <<<"$out"; then
  printf '  %-38s DID NOT BUILD (inconclusive)\n' "$label"
elif grep -q "test result: FAILED" <<<"$out"; then
  printf '  %-38s RED (good)\n' "$label"
elif grep -qE "^test result: ok\. 0 passed" <<<"$out"; then
  printf '  %-38s NO TEST MATCHED "%s"\n' "$label" "$filter"
elif grep -qi "skipping" <<<"$out"; then
  # This machine lacks what the test needs — no GPU adapter, no power backend. It keys on the
  # convention every hardware-gated test in this tree already follows: print what is missing,
  # ending in "skipping", and return. NOT the same as a blind test, and NOT a pass. The caller
  # decides what to do about it, and `FERROTHERM_REQUIRE_ALL=1` makes it fatal.
  # The WHOLE line, not the fragment after the last semicolon. A first cut matched `[^;]*skipping`,
  # which cannot cross a `;` and so reported the reason as the bare word "skipping" — dropping
  # "no GPU adapter on this machine", which is the only part a reader needs.
  printf '  %-38s NOT EVALUATED HERE (%s)\n' "$label" \
    "$(grep -i 'skipping' <<<"$out" | head -1 | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
else
  printf '  %-38s STILL GREEN — the test is blind\n' "$label"
fi
