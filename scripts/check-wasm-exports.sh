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

# The README states a size, and a stated number drifts.
#
# It said 44 KB while the artefact was 356 KB -- eight times out, and nothing could see it because
# the figure lived in prose and the artefact lived in a file. Corrected once by hand, it went stale
# again within a day when the OMMX rewrite added a module. A number a human retypes is a number that
# rots, so the check reads both.
#
# 10% of slack, because a wasm is not byte-reproducible across toolchain versions and a gate that
# fires on a rustc bump is a gate people disable.
readme="$here/README.md"
if [[ -f "$readme" ]]; then
  raw_kb=$(( $(wc -c < "$wasm") / 1024 ))
  gz_kb=$(( $(gzip -c "$wasm" | wc -c) / 1024 ))
  claimed=$(grep -oE '\*\*[0-9]+ KB \.wasm\*\*' "$readme" | grep -oE '[0-9]+' | head -1)
  claimed_gz=$(grep -oE '\([0-9]+ KB gzipped\)' "$readme" | grep -oE '[0-9]+' | head -1)
  if [[ -z "$claimed" ]]; then
    echo "the README no longer states a wasm size; this check now guards nothing" >&2
    exit 2
  fi
  bad_size=0
  for pair in "$claimed:$raw_kb:raw" "$claimed_gz:$gz_kb:gzipped"; do
    said=${pair%%:*}; rest=${pair#*:}; actual=${rest%%:*}; what=${rest##*:}
    [[ -n "$said" ]] || continue
    lo=$(( actual * 9 / 10 )); hi=$(( actual * 11 / 10 ))
    if (( said < lo || said > hi )); then
      echo "README says ${said} KB ${what}; the artefact is ${actual} KB" >&2
      bad_size=1
    fi
  done
  if (( bad_size )); then
    echo "A size a human retypes is a size that rots. Update README.md, or rebuild docs/*.wasm." >&2
    exit 1
  fi
  # AND THE ARTEFACT MUST BE STRIPPED, which the size figures above cannot enforce: they carry a
  # +/-10% band so a human-retyped number does not rot on every rebuild, and an unstripped binary is
  # inside that band. It costs 62 KB raw and 12 KB gzipped on every page load, forever, for a name
  # section nobody reads -- the browser's own devtools are the only consumer, and a developer
  # debugging the sampler rebuilds without the flag anyway.
  if ! python3 - "$wasm" <<'PYEOF'
import sys

def uleb(b, i):
    v = s = 0
    while True:
        c = b[i]; i += 1
        v |= (c & 0x7F) << s
        if not c & 0x80:
            return v, i
        s += 7

b = open(sys.argv[1], "rb").read()
if b[:4] != b"\0asm":
    print("not a wasm binary", file=sys.stderr); sys.exit(2)
i, found = 8, []
while i < len(b):
    sid = b[i]; i += 1
    size, i = uleb(b, i)
    if sid == 0:                      # custom section: payload begins with its own name
        n, j = uleb(b, i)
        found.append(b[j:j + n].decode("utf-8", "replace"))
    i += size
bad = [n for n in found if n in ("name", ".debug_info", ".debug_line")]
if bad:
    print("carries " + ", ".join(repr(n) for n in bad) + " -- build it with "
          "RUSTFLAGS='-C strip=symbols'", file=sys.stderr)
    sys.exit(1)
PYEOF
  then
    echo "the committed wasm was not stripped, which costs every visitor bytes for nothing" >&2
    exit 1
  fi
  printf 'README size figures agree: %s KB raw, %s KB gzipped (stripped)\n' "$raw_kb" "$gz_kb"
fi

if [[ ${#missing[@]} -gt 0 ]]; then
  printf '\nmissing: %s\n' "${missing[*]}" >&2
  echo >&2
  echo "The page will load and the call will be undefined, which is silent. Rebuild:" >&2
  echo "  cargo build --release --lib --target wasm32-unknown-unknown" >&2
  echo "  cp target/wasm32-unknown-unknown/release/ferrotherm.wasm docs/" >&2
  exit 1
fi
echo "all present"
