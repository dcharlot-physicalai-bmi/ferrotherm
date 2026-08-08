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

missing=()
for sym in "${wanted[@]}"; do
  grep -q "$sym" "$wasm" || missing+=("$sym")
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
