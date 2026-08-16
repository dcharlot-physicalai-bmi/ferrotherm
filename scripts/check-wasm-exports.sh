#!/usr/bin/env bash
#
# Does the committed wasm export everything the pages call?
#
# The requirement is derived from the pages themselves -- every `W.ft_...` in docs/*.html -- rather
# than from a list kept by hand, because a hand-kept list drifts the moment someone adds a call.
#
# This is deliberately NOT a byte comparison against a fresh build. A wasm binary is not
# reproducible across toolchain versions, so comparing bytes fails on a CI runner whose rustc
# differs from the author's, which says nothing about whether the artefact works. What matters is
# that every symbol the page reaches for is there: a missing export makes the call `undefined`, and
# calling undefined in a click handler leaves the page looking fine and doing nothing.
#
#   scripts/check-wasm-exports.sh [path/to/ferrotherm.wasm]

set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
wasm="${1:-$here/docs/ferrotherm.wasm}"

if [[ ! -f "$wasm" ]]; then
  echo "no wasm at $wasm" >&2
  exit 2
fi

# every ft_* symbol reached through the wasm exports object in any page
# (read into an array without mapfile, which macOS's bash 3.2 does not have)
wanted=()
while IFS= read -r sym; do
  [[ -n "$sym" ]] && wanted+=("$sym")
done < <(grep -ohE '\bW\.(ft_[a-z0-9_]+)' "$here"/docs/*.html | sed 's/^W\.//' | sort -u)

if [[ ${#wanted[@]} -eq 0 ]]; then
  echo "found no ft_* calls in docs/*.html, which is itself suspicious" >&2
  exit 2
fi

# Matched against the wasm EXPORT SECTION, not as a substring of the whole binary.
#
# `grep -q "$sym" "$wasm"` was true if the name appeared anywhere at all -- inside a longer symbol
# (`ft_len` inside `ft_length`), in a debug string, in a data segment. Measured on the shipped
# artefact: 11 of the 77 names passed on a substring hit rather than a real export, so the gate
# would have stayed green through a rename that broke every page.
#
# In a wasm binary an export name is stored length-prefixed, and for names under 128 bytes the
# prefix is a single byte equal to the length. Matching `<len><name>` is exact enough to reject a
# longer symbol that merely contains the one we want, and needs no wasm parser.
missing=()
for sym in "${wanted[@]}"; do
  python3 - "$wasm" "$sym" <<'PY_MATCH' || missing+=("$sym")
import sys
blob = open(sys.argv[1], "rb").read()
name = sys.argv[2].encode()
# A name shorter than 128 bytes is preceded by one byte holding its length.
sys.exit(0 if bytes([len(name)]) + name in blob else 1)
PY_MATCH
done

printf 'checked %d exports the pages call against %s (%s bytes)\n' \
  "${#wanted[@]}" "$(basename "$wasm")" "$(wc -c < "$wasm" | tr -d ' ')"

if [[ ${#missing[@]} -gt 0 ]]; then
  printf '\nmissing: %s\n' "${missing[*]}" >&2
  echo >&2
  echo "The page will load and the call will be undefined, which is silent. Rebuild:" >&2
  echo "  cargo build --release --lib --target wasm32-unknown-unknown" >&2
  echo "  cp target/wasm32-unknown-unknown/release/ferrotherm.wasm docs/" >&2
  exit 1
fi
echo "all present"
