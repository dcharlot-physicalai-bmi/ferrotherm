#!/usr/bin/env bash
# One mutation against a file that is NOT YET COMMITTED.
#
#   scripts/mutation-hand.sh <file> <old> <new> <test-filter> <label> [package]
#
# `mutation-check.sh` restores with git, so it refuses a file with uncommitted changes -- which is
# every file at the moment its tests most need to be shown able to fail: when it is being written.
# This is the same check for that moment. It lived in a scratch directory through three sessions
# and was wiped twice; it is here so the fourth does not rewrite it from memory.
#
# THREE THINGS IT DOES BECAUSE THEIR ABSENCE EACH LIED ONCE (2026-09-19):
#
# 1. RESTORES FROM SAVED BYTES, never by reverse-replace. A mutant whose NEW text already occurred
#    earlier in the file was "restored" at the wrong site, and the measured flip energies of two
#    boards sat swapped in `ledger.rs` until a checksum refused to continue.
# 2. REFUSES a mutant whose new text is already in the file, for the same reason.
# 3. TAKES ITS VERDICT FROM THE `test result:` LINE. A failing test run prints `error: test failed`
#    at column zero, so reading `^error` as "did not build" called three caught mutants broken.
#
# The file's sha256 is compared before and after, always; a mismatch exits 9 and says so.
# Arguments may contain newlines. A label is free text. Exit 0 whatever the verdict: the verdict is
# the printed line, and a STILL GREEN is a finding, not a failure of this script.

set -uo pipefail
if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 <file> <old> <new> <test-filter> <label> [package]" >&2
  exit 2
fi
file="$1"; old="$2"; new="$3"; filter="$4"; label="$5"; pkg="${6:-}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
base="$(sha "$file")"
keep="$(mktemp)"
python3 -c "import sys;open(sys.argv[2],'wb').write(open(sys.argv[1],'rb').read())" "$file" "$keep"
restore() {
  python3 -c "import sys;open(sys.argv[1],'wb').write(open(sys.argv[2],'rb').read())" "$file" "$keep"
}
trap 'restore; rm -f "$keep"' EXIT INT TERM HUP

python3 - "$file" "$old" "$new" <<'PY' || { printf '  %-44s DID NOT APPLY (old absent or not unique, or new text already present)\n' "$label"; exit 0; }
import sys
path, old, new = sys.argv[1:4]
t = open(path).read()
if t.count(old) != 1 or new in t:
    sys.exit(1)
open(path, 'w').write(t.replace(old, new, 1))
PY

if [[ -n "$pkg" ]]; then
  out="$(cargo test --release --lib -p "$pkg" "$filter" 2>&1)"
else
  out="$(cargo test --release --lib "$filter" 2>&1)"
fi
restore
if [[ "$(sha "$file")" != "$base" ]]; then
  echo "RESTORE FAILED for '$label': $file is not the file this started with" >&2
  exit 9
fi

line="$(grep -E '^test result:' <<<"$out" | head -1)"
if [[ -z "$line" ]]; then
  printf '  %-44s NO TEST RESULT LINE (did not build, or did not run):\n' "$label"
  tail -6 <<<"$out"
elif grep -q 'FAILED' <<<"$line"; then
  printf '  %-44s RED (good)\n' "$label"
elif grep -q ' 0 passed' <<<"$line"; then
  printf '  %-44s NO TEST MATCHED\n' "$label"
else
  printf '  %-44s STILL GREEN -- the test is blind to this\n' "$label"
fi
