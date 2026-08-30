#!/usr/bin/env bash
#
# Do the bindings solve one HIGHER-ORDER model to the same answer?
#
# `check-answers.sh` puts one model through every surface and compares the answer. Its model is
# pairwise, because until now the higher-order path reached exactly one surface -- Rust -- and there
# was nothing to compare. `ft_hubo_*` changed that, and a new family of twenty-odd entry points
# across four bindings is precisely the shape of thing that compiles everywhere and computes the
# same number nowhere.
#
# THE MODEL, and why this one. A three-body parity term over three spins:
#
#     E(s) = -s0 * s1 * s2,  minimised at -1 when the product is +1
#
# It is the smallest model that CANNOT be expressed pairwise at all, so a surface that quietly
# routed it through the reduction would answer with ancillas and be caught by the ancilla count. Its
# optimum is -1 with a four-fold degenerate state, so the ENERGY is compared and the state is
# checked only for the invariant that decides it -- the product being +1 -- rather than for one
# particular assignment two correct bindings are entitled to disagree about.
#
# Also compared: terms, max arity, and ancillas_avoided. The last is the module's whole measurable
# claim, and a binding that reported it wrong would be reporting the wrong saving for the right
# answer, which no energy comparison can see.
#
#   scripts/check-hubo-answers.sh              every surface answers the same higher-order model
#   scripts/check-hubo-answers.sh --selftest   prove the comparison can fail
#
# A missing toolchain SKIPS; a toolchain that is present and produces nothing FAILS.
# FERROTHERM_REQUIRE_ALL=1 turns every skip into a refusal, and CI sets it.

set -uo pipefail
self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

# Where the Python surface is imported from. Always the tree's own `python/`, except under
# --selftest, which points it at a DAMAGED copy in a temp dir so this gate can be run against a
# binding that is wrong on purpose. Nothing in the tree is ever edited.
PYDIR="${FERROTHERM_SELFTEST_PYDIR:-python}"

# ---- selftest -----------------------------------------------------------------------------------
#
# Run this whole gate against a binding that is WRONG ON PURPOSE, and require it to say so. A gate
# nobody has watched fail is not a gate: `check-versions.sh` reported "all 54 crates" for a
# repository with six for as long as anyone had looked, and it looked green throughout.
#
# WHY THESE TWO DAMAGES. This gate exists because a new family of higher-order entry points landed
# on four bindings at once, and the way a family like that goes wrong is not that a surface stops
# working -- it is that one surface reads the SAME C ABI slightly differently to the others. So both
# damages are one binding misreading one thing, which is the mistake that actually gets made:
#
#   1. THE WEIGHT SIGN. `add([0,1,2], 1.0)` hands the weight straight to ft_hubo_add; a binding that
#      negated it -- an off-by-a-minus in a marshalling layer, or a surface written against the
#      opposite energy convention -- would still anneal, still converge, and still report energy=-1,
#      because -s0*s1*s2 and +s0*s1*s2 have the same optimum. It shows up ONLY in the product being
#      -1 instead of +1. That is exactly why the header says the state is checked for the invariant
#      that decides it rather than only for the energy, and this damage is what proves the invariant
#      load-bearing: an energy-only comparison would wave it straight through.
#
#   2. THE TERM ARITY. The accessors sit one line apart in the ctypes signature table, and each is a
#      name bound to a string. Wiring ft_hubo_max_arity to ft_hubo_terms is a copy-paste away and
#      raises nothing at import, at call, or in any test that does not compare the number against
#      another surface -- both are u32, both are small, and on the one-term one-variable model a
#      unit test reaches for first they are the same number. This is the "compiles everywhere,
#      computes the same number nowhere" shape the gate was written for.
#
# Both are applied to a COPY of python/ in a temp dir, and only the Python surface's import path
# moves; nothing in the tree is written, and the tree's own comparison logic runs untouched.
if [ "${1:-}" = "--selftest" ]; then
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT

  # The binding finds the shared library at ../../target/release relative to its own package
  # directory, so a copy sitting in a temp dir would fail to LOAD rather than answer wrongly -- and
  # a selftest that reads that as "caught" is testing dlopen, not the comparison. This is not
  # hypothetical: it is what the first run of this selftest actually did, and only the "the gate
  # must name the damaged surface" guard below told the difference. One symlink puts the copy back
  # in the same relationship to the build directory that the tree's own package has. rm -rf unlinks
  # a symlink rather than descending it, so the trap above cannot reach target/.
  ln -s "$here/target" "$tmp/target"

  # damage <label> <sed expression> <text the damage must leave behind> <what it stands for>
  damage() {
    rm -rf "$tmp/python"
    cp -R python "$tmp/python" || { echo "SELFTEST FAILED: could not copy the binding" >&2; return 1; }
    rm -rf "$tmp/python/ferrotherm/__pycache__"
    sed "$2" python/ferrotherm/__init__.py > "$tmp/python/ferrotherm/__init__.py"

    # Prove the damage LANDED. A sed that matched nothing leaves a pristine copy, the gate passes,
    # and reading that pass as "the damage was caught" is the vacuous selftest this guards against.
    if ! grep -qF -- "$3" "$tmp/python/ferrotherm/__init__.py"; then
      echo "SELFTEST FAILED: the damage did not apply ($1) -- the binding no longer looks the way" >&2
      echo "this expects, so nothing was tested. Fix the sed, not the gate." >&2
      return 1
    fi
    if diff -q python/ferrotherm/__init__.py "$tmp/python/ferrotherm/__init__.py" >/dev/null 2>&1; then
      echo "SELFTEST FAILED: the damage did not apply ($1) -- the copy is identical to the tree" >&2
      return 1
    fi

    FERROTHERM_SELFTEST_PYDIR="$tmp/python" bash "$self" > "$tmp/$1.log" 2>&1
    st=$?
    if [ "$st" -eq 0 ]; then
      echo "SELFTEST FAILED: $4 slipped through -- the gate still reported agreement" >&2
      sed 's/^/      /' "$tmp/$1.log" >&2
      return 1
    fi
    # Non-zero is not enough. A missing toolchain, a failed build or a surface that produced nothing
    # also exits non-zero, and any of those would leave the real disagreement unmeasured while this
    # printed a tick. Require that the DAMAGED surface is the one the comparison named.
    if ! grep -q '^  python .*<- expected' "$tmp/$1.log"; then
      echo "SELFTEST FAILED: the gate exited $st without ever naming the python surface, so it fell" >&2
      echo "over for another reason and $4 was never actually compared" >&2
      sed 's/^/      /' "$tmp/$1.log" >&2
      return 1
    fi
    printf '  caught: %s\n' "$4"
    return 0
  }

  echo
  fail=0
  damage sign 's/_hubo_add(self\._h, float(weight))/_hubo_add(self._h, -float(weight))/' \
    '_hubo_add(self._h, -float(weight))' \
    'one binding negating the weight it was handed -- invisible to the energy, caught by the product' \
    || fail=1
  damage arity 's/_hubo_max_arity = _sig("ft_hubo_max_arity"/_hubo_max_arity = _sig("ft_hubo_terms"/' \
    '_hubo_max_arity = _sig("ft_hubo_terms"' \
    'one binding reading the term arity off the wrong C symbol' \
    || fail=1

  echo
  [ "$fail" -eq 0 ] || exit 1
  echo "selftest: a flipped weight sign and a misread arity were both caught, so this gate can fail"
  exit 0
fi

out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT

cargo build --release --quiet --lib 2>/dev/null || { echo "the library did not build" >&2; exit 2; }
LIB="$here/target/release/libferrotherm.dylib"
[ -f "$LIB" ] || LIB="$here/target/release/libferrotherm.so"
[ -f "$LIB" ] || { echo "no cdylib after building it" >&2; exit 2; }

require_all() { [ "${FERROTHERM_REQUIRE_ALL:-0}" = "1" ]; }
missing=""
say()  { printf '  %-10s %s\n' "$1" "$2"; }
skip() { missing="$missing $1"; printf '  %-10s skipped: %s\n' "$1" "$2"; }

EXPECT="energy=-1 product=1 terms=1 arity=3 ancillas=1"
attempted=""
attempt() { attempted="$attempted $1"; }

# ---- rust: the reference ---------------------------------------------------------------------
mkdir -p examples
cat > examples/_hans.rs <<'RS'
fn main() {
    use ferrotherm::hubo::{anneal, Hubo, Params};
    let mut h = Hubo::new(3);
    h.add(&[0, 1, 2], 1.0).unwrap();
    let o = anneal(&h, &Params::default(), 7);
    let p = o.state[0] as i32 * o.state[1] as i32 * o.state[2] as i32;
    print!(
        "energy={} product={p} terms={} arity={} ancillas={}",
        o.energy, h.terms(), h.max_arity(), h.ancillas_avoided()
    );
}
RS
attempt rust
cargo run --release --quiet --example _hans > "$out/rust.txt" 2>"$out/rust.err"
rm -f examples/_hans.rs

# ---- python -----------------------------------------------------------------------------------
attempt python
python3 - "$out" "$PYDIR" 2> "$out/python.err" <<'PY'
import sys, os
sys.path.insert(0, sys.argv[2])
import ferrotherm as ft
h = ft.Hubo(3)
h.add([0, 1, 2], 1.0)
e = h.anneal(seed=7)
s = h.state
open(os.path.join(sys.argv[1], "python.txt"), "w").write(
    f"energy={e:g} product={s[0]*s[1]*s[2]} terms={h.terms} "
    f"arity={h.max_arity} ancillas={h.ancillas_avoided}")
PY

# ---- julia ------------------------------------------------------------------------------------
if command -v julia >/dev/null 2>&1; then
  attempt julia
  FERROTHERM_LIB="$LIB" julia --project=julia/Ferrotherm -e '
    using Ferrotherm
    h = Hubo(3)
    add!(h, [1, 2, 3], 1.0)
    e = anneal!(h; seed = 7)
    s = spins(h)
    print("energy=", Int(e), " product=", Int(s[1]) * Int(s[2]) * Int(s[3]),
          " terms=", terms(h), " arity=", max_arity(h), " ancillas=", ancillas_avoided(h))
  ' > "$out/julia.txt" 2> "$out/julia.err"
else skip julia "no julia on PATH"; fi

# ---- zig --------------------------------------------------------------------------------------
if command -v zig >/dev/null 2>&1; then
  attempt zig
  cat > zig/_hans.zig <<'ZG'
const std = @import("std");
const ft = @import("ferrotherm.zig");
pub fn main() !void {
    const h = try ft.Hubo.init(3);
    defer h.deinit();
    try h.add(&.{ 0, 1, 2 }, 1.0);
    const e = try h.anneal(0, 0, 0, 0, 7);
    var s: [3]i8 = undefined;
    try h.read(&s);
    std.debug.print("energy={d} product={d} terms={d} arity={d} ancillas={d}", .{
        @as(i64, @intFromFloat(e)), @as(i32, s[0]) * s[1] * s[2],
        h.terms(), h.maxArity(), h.ancillasAvoided(),
    });
}
ZG
  (cd zig && zig run _hans.zig -I "$here/include" -L "$here/target/release" -lferrotherm -lc \
      2> "$out/zig.raw" >/dev/null)
  if [ $? -eq 0 ]; then cp "$out/zig.raw" "$out/zig.txt"
  else head -1 "$out/zig.raw" > "$out/zig.err"; : > "$out/zig.txt"; fi
  rm -f zig/_hans.zig
else skip zig "no zig on PATH"; fi

# ---- http and mcp -------------------------------------------------------------------------------
#
# Two more surfaces, and the two an agent actually reaches for. They share one dispatch, so they
# cannot disagree with each other -- but they can both disagree with the library, which is what a
# comparison against the other four catches.
if cargo build --release --quiet -p ferrotherm-serve 2>/dev/null; then
  BODY='{"spins":3,"terms":[{"vars":[0,1,2],"weight":1.0}],"seed":7}'
  # Single quotes throughout the Python, so nothing here needs a backslash inside an f-string --
  # which Python rejects outright and which cost this gate one run to discover.
  READ='
import json, sys
d = json.load(sys.stdin)
s = d["state"]
prod = int(s[0]) * int(s[1]) * int(s[2])
print("energy=%g product=%d terms=%d arity=%d ancillas=%d"
      % (d["energy"], prod, d["terms"], d["max_arity"], d["ancillas_avoided"]), end="")'

  attempt http
  (./target/release/ferrotherm-serve 127.0.0.1:8487 >/dev/null 2>&1 &)
  for _ in $(seq 1 40); do
    curl -s -o /dev/null -X POST localhost:8487/v1/capabilities -d '{}' 2>/dev/null && break
    sleep 0.25
  done
  curl -s -X POST localhost:8487/v1/hubo -d "$BODY" 2>/dev/null \
    | python3 -c "$READ" > "$out/http.txt" 2> "$out/http.err"
  pkill -f 'ferrotherm-serve 127.0.0.1:8487' 2>/dev/null

  attempt mcp
  printf '%s\n' "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"ferrotherm_hubo\",\"arguments\":$BODY}}" \
    | ./target/release/ferrotherm-mcp 2>/dev/null \
    | python3 -c "
import json,sys
line = next(l for l in sys.stdin if l.strip())
print(json.loads(line)['result']['content'][0]['text'], end='')" 2>/dev/null \
    | python3 -c "$READ" > "$out/mcp.txt" 2> "$out/mcp.err"
else
  skip http "ferrotherm-serve did not build"
  skip mcp "ferrotherm-serve did not build"
fi

# ---- compare ----------------------------------------------------------------------------------
echo
bad=0
n=0
for name in $attempted; do
  f="$out/$name.txt"
  got=""
  [ -f "$f" ] && got=$(tr -d '\r\n' < "$f" 2>/dev/null)
  if [ -z "$got" ]; then
    say "$name" "PRODUCED NOTHING -- its toolchain is present, so it broke"
    [ -s "$out/$name.err" ] && sed 's/^/      /' "$out/$name.err" | tail -4
    bad=$((bad + 1)); n=$((n + 1)); continue
  fi
  n=$((n + 1))
  if [ "$got" = "$EXPECT" ]; then say "$name" "$got"
  else say "$name" "$got   <- expected $EXPECT"; bad=$((bad + 1)); fi
done

echo
if [ "$n" -lt 2 ]; then
  echo "only $n surface(s) answered; this compared nothing" >&2
  exit 2
fi
if [ "$bad" -gt 0 ]; then
  echo "$bad of $n surfaces did not return the agreed answer on a model that cannot be" >&2
  echo "expressed pairwise at all. A surface routing it through the reduction instead would" >&2
  echo "answer with ancillas, which is why the ancilla count is in the comparison." >&2
  exit 1
fi
if require_all && [ -n "$missing" ]; then
  echo "FERROTHERM_REQUIRE_ALL=1 and these surfaces were skipped:${missing}" >&2
  exit 2
fi
echo "  $n surfaces solve one higher-order model to the same answer, with no ancillas"
