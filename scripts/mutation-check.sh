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
if [[ -n "$pkg" ]]; then
  out="$(cargo test --release --lib -p "$pkg" "$filter" 2>&1)"
else
  out="$(cargo test --release --lib "$filter" 2>&1)"
fi
git checkout -- "$file"

if ! grep -q "^test result:" <<<"$out"; then
  printf '  %-38s DID NOT BUILD (inconclusive)\n' "$label"
elif grep -q "test result: FAILED" <<<"$out"; then
  printf '  %-38s RED (good)\n' "$label"
elif grep -qE "^test result: ok\. 0 passed" <<<"$out"; then
  printf '  %-38s NO TEST MATCHED "%s"\n' "$label" "$filter"
else
  printf '  %-38s STILL GREEN — the test is blind\n' "$label"
fi
