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
#   scripts/check-hubo-answers.sh
#
# A missing toolchain SKIPS; a toolchain that is present and produces nothing FAILS.
# FERROTHERM_REQUIRE_ALL=1 turns every skip into a refusal, and CI sets it.

set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
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
python3 - "$out" 2> "$out/python.err" <<'PY'
import sys, os
sys.path.insert(0, "python")
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
