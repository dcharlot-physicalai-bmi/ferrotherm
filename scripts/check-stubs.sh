#!/usr/bin/env bash
# The type stub must describe the library that exists, not the one it described last month.
#
# A stub is the surface an editor, a type checker and any model writing code against this package
# actually read. A hand-kept one drifts a signature at a time and every drift is a confident lie
# that still autocompletes -- so this regenerates from the runtime API and requires the committed
# file to match byte for byte. Renaming a parameter without regenerating fails here.
#
#   scripts/check-stubs.sh              the committed stub matches the library
#   scripts/check-stubs.sh --selftest   prove the comparison can fail
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

stub="python/ferrotherm/__init__.pyi"
marker="python/ferrotherm/py.typed"

if [ ! -f "$marker" ]; then
  echo "missing $marker: without the PEP 561 marker a type checker ignores the stub entirely," >&2
  echo "which is the same as shipping no stub at all" >&2
  exit 1
fi
if [ ! -f "$stub" ]; then
  echo "missing $stub; run: scripts/gen-stubs.py" >&2
  exit 1
fi

fresh="$(mktemp)"
trap 'rm -f "$fresh"' EXIT
python3 scripts/gen-stubs.py --stdout > "$fresh"

if [ "${1:-}" = "--selftest" ]; then
  # Damage a copy the way real drift looks -- one renamed parameter -- and require the diff to see
  # it. A check that cannot fail on its first run is the same evidence as no check.
  damaged="$(mktemp)"
  trap 'rm -f "$fresh" "$damaged"' EXIT
  sed 's/def rbm(visible: int/def rbm(visable: int/' "$stub" > "$damaged"
  if cmp -s "$damaged" "$fresh"; then
    echo "SELFTEST FAILED: a renamed parameter compared EQUAL, so this check proves nothing" >&2
    exit 1
  fi
  if ! grep -q "visable" "$damaged"; then
    echo "SELFTEST FAILED: the damage did not apply, so the comparison above was vacuous" >&2
    exit 1
  fi
  echo "selftest: a renamed parameter was caught, so the comparison can fail"
  exit 0
fi

if ! diff -u "$stub" "$fresh" > /tmp/stub.diff 2>&1; then
  echo "$stub does not match the library it describes:" >&2
  head -40 /tmp/stub.diff >&2
  echo >&2
  echo "regenerate it with: scripts/gen-stubs.py" >&2
  exit 1
fi

lines=$(wc -l < "$stub" | tr -d ' ')
syms=$(grep -cE '^(class |def )' "$stub" || true)
echo "the stub matches the library: $syms top-level names over $lines lines, and py.typed is present"
