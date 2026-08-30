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
#   scripts/check-wasm-exports.sh --selftest      prove both checks can fail

set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --selftest runs this gate against a DAMAGED COPY of the tree and demands a failure, twice, once
# for each thing the gate claims to guard.
#
# It exists because both halves of this check are silent when they break. A page that calls a
# symbol the wasm does not export still loads, still renders, and does nothing when clicked; an
# unstripped binary is a working binary that quietly costs every visitor bytes. Neither shows up as
# an error anywhere, which is exactly the condition under which a green tick has to be earned
# rather than assumed -- and this gate had already been wrong once in the direction a passing run
# cannot show: it matched export names as bare substrings, so 11 of 77 names passed on a hit inside
# a longer symbol and the gate would have stayed green through a rename that broke every page.
#
# THE TWO DAMAGES, and why these:
#
#   1. One call site in docs/ide.html renamed to a symbol the wasm does not export. This is the
#      regression this gate was written for and the one that actually happens: a symbol is renamed
#      or split in src/ffi.rs, the page is updated to the new name, and the committed wasm is not
#      rebuilt (or vice versa). Renaming the CALL rather than deleting it is the honest shape --
#      the page still parses, the handler still runs, and W.ft_sweep_batch is `undefined`.
#
#   2. A `name` custom section appended to a copy of the wasm -- what a build without
#      RUSTFLAGS='-C strip=symbols' leaves behind. Deliberately a SMALL one, a few KB of mangled
#      Rust names rather than the full 62 KB, because the point is the claim in the comment below:
#      the README size band is +/-10% and an unstripped binary fits inside it, so the size figures
#      cannot enforce stripping and the section walk must. A 62 KB section would trip the size
#      check first and prove the wrong thing.
#
# Both run the gate from a mktemp copy. Nothing in the repository is written to, transiently or
# otherwise.
if [[ "${1:-}" == "--selftest" ]]; then
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT
  mkdir -p "$tmp/scripts" "$tmp/docs"
  cp "$here/scripts/check-wasm-exports.sh" "$tmp/scripts/"
  cp "$here"/docs/*.html "$tmp/docs/"
  cp "$here/docs/ferrotherm.wasm" "$tmp/docs/"
  if [[ -f "$here/README.md" ]]; then cp "$here/README.md" "$tmp/"; fi

  # Positive control. If the copy itself cannot pass, a failure below says nothing about the
  # damage -- it says the harness is broken, which is the way a self-test lies.
  if ! ( cd "$tmp" && bash scripts/check-wasm-exports.sh ) >"$tmp/control.out" 2>&1; then
    echo "SELFTEST FAILED: the undamaged copy does not pass, so no failure below is attributable:" >&2
    sed 's/^/  /' "$tmp/control.out" >&2
    exit 1
  fi

  # ---- damage 1: a page calling a symbol that does not exist ------------------------------------
  sed 's/W\.ft_sweep(/W.ft_sweep_batch(/g' "$here/docs/ide.html" > "$tmp/docs/ide.html"
  if ! grep -q 'W\.ft_sweep_batch(' "$tmp/docs/ide.html"; then
    echo "SELFTEST FAILED: the damage did not apply -- docs/ide.html has no W.ft_sweep( call to rename" >&2
    exit 1
  fi
  if diff -q "$here/docs/ide.html" "$tmp/docs/ide.html" >/dev/null; then
    echo "SELFTEST FAILED: the damage did not apply -- the damaged page is identical to the original" >&2
    exit 1
  fi
  # And the name has to be genuinely absent from the wasm, or the gate is right to pass it.
  if python3 -c 'import sys; b=open(sys.argv[1],"rb").read(); n=b"ft_sweep_batch"; sys.exit(0 if bytes([len(n)])+n in b else 1)' "$here/docs/ferrotherm.wasm"; then
    echo "SELFTEST FAILED: the wasm exports ft_sweep_batch after all, so this damage is not damage" >&2
    exit 1
  fi
  if ( cd "$tmp" && bash scripts/check-wasm-exports.sh ) >"$tmp/missing.out" 2>&1; then
    echo "SELFTEST FAILED: the gate passed a page calling W.ft_sweep_batch, which the wasm does not export" >&2
    exit 1
  fi
  if ! grep -q 'missing: .*ft_sweep_batch' "$tmp/missing.out"; then
    echo "SELFTEST FAILED: the gate failed, but not on the missing export -- it said:" >&2
    sed 's/^/  /' "$tmp/missing.out" >&2
    exit 1
  fi
  cp "$here"/docs/*.html "$tmp/docs/"   # undamage the pages before the second run

  # ---- damage 2: an unstripped binary ------------------------------------------------------------
  if ! python3 - "$here/docs/ferrotherm.wasm" "$tmp/unstripped.wasm" <<'PY_UNSTRIP'
import sys

def uleb(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)

def nstr(s):
    b = s.encode()
    return uleb(len(b)) + b

src, dst = sys.argv[1], sys.argv[2]
blob = open(src, "rb").read()

# A real name section: subsection 0 is the module name, subsection 1 a vector of function names.
# The names are the mangled shape rustc emits, so what the walker finds looks like what a
# non-stripping build leaves, not like a marker planted for a test.
mod = uleb(0) + uleb(len(nstr("ferrotherm"))) + nstr("ferrotherm")
entries = bytearray(uleb(200))
for i in range(200):
    entries += uleb(i) + nstr("_ZN10ferrotherm7sampler5sweep17h%016xE" % (0x5eed0000 + i))
fns = uleb(1) + uleb(len(entries)) + bytes(entries)
payload = nstr("name") + mod + fns
section = b"\x00" + uleb(len(payload)) + payload
open(dst, "wb").write(blob + section)
PY_UNSTRIP
  then
    echo "SELFTEST FAILED: the damage did not apply -- could not build an unstripped copy" >&2
    exit 1
  fi
  # Prove the damage landed, by the same reading the gate itself does: a custom section named
  # "name", with every original byte untouched ahead of it.
  if ! python3 - "$here/docs/ferrotherm.wasm" "$tmp/unstripped.wasm" <<'PY_ASSERT'
import sys

def uleb(b, i):
    v = s = 0
    while True:
        c = b[i]; i += 1
        v |= (c & 0x7F) << s
        if not c & 0x80:
            return v, i
        s += 7

orig = open(sys.argv[1], "rb").read()
dmg = open(sys.argv[2], "rb").read()
if dmg[:len(orig)] != orig:
    print("the damaged copy changed bytes of the original module", file=sys.stderr); sys.exit(1)
i, names = 8, []
while i < len(dmg):
    sid = dmg[i]; i += 1
    size, i = uleb(dmg, i)
    if sid == 0:
        n, j = uleb(dmg, i)
        names.append(dmg[j:j + n].decode("utf-8", "replace"))
    i += size
if "name" not in names:
    print("no custom section called name in the damaged copy: %r" % (names,), file=sys.stderr)
    sys.exit(1)
PY_ASSERT
  then
    echo "SELFTEST FAILED: the damage did not apply -- the copy carries no name section" >&2
    exit 1
  fi
  if ( cd "$tmp" && bash scripts/check-wasm-exports.sh "$tmp/unstripped.wasm" ) >"$tmp/strip.out" 2>&1; then
    echo "SELFTEST FAILED: the gate passed a wasm carrying a name section, so nothing enforces stripping" >&2
    exit 1
  fi
  if ! grep -q 'was not stripped' "$tmp/strip.out"; then
    # A size failure here would mean the injected section was big enough to trip the README band
    # instead, which proves the size check and not this one.
    echo "SELFTEST FAILED: the gate rejected the unstripped binary for another reason -- it said:" >&2
    sed 's/^/  /' "$tmp/strip.out" >&2
    exit 1
  fi

  echo "selftest: a page calling an unexported symbol was caught, and so was an unstripped wasm"
  exit 0
fi
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
