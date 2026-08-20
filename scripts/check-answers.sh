#!/usr/bin/env bash
#
# Do the bindings SOLVE the same, or only BUILD the same?
#
# `check-parity.sh` proves every C ABI symbol reaches every binding. `check-semantics.sh` proves each
# one compiles a model to byte-identical `.ftp`. Both were green while Zig's `Problem.solve` failed
# to solve at all: it never called `ft_model_compile`, where Python and Julia both do, so the same
# program written from their examples returned `Error.NotSolved` -- the sampler reporting failure
# when nothing had been built for it to sample.
#
# Neither existing gate could see it. Parity checks NAMES. Semantics checks the model BEFORE it is
# solved -- and that path calls `compile()` explicitly, so it exercised the very step `solve` was
# skipping. The gap between "builds the same model" and "returns the same answer" is exactly one
# function call wide, and a bug lived in it.
#
# So: one model, deliberately with a UNIQUE optimum, solved end to end through every surface. Same
# answer or the surface is wrong.
#
#   scripts/check-answers.sh
#
# A missing toolchain SKIPS. A toolchain that is present and produces nothing FAILS -- because that
# means the binding broke, which is the distinction `check-semantics.sh` learned the hard way.

set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
out=$(mktemp -d)
cleanup() { rm -rf "$out"; }
trap cleanup EXIT

# Build the cdylib the bindings dlopen. `cargo run --example` links an rlib and does not emit it, so
# on a clean runner nothing would exist to load -- the defect that kept CI red for weeks.
cargo build --release --quiet --lib 2>/dev/null || { echo "the library did not build" >&2; exit 2; }
LIB="$here/target/release/libferrotherm.dylib"
[ -f "$LIB" ] || LIB="$here/target/release/libferrotherm.so"
[ -f "$LIB" ] || { echo "no cdylib after building it" >&2; exit 2; }

say()  { printf '  %-10s %s\n' "$1" "$2"; }
skip() { printf '  %-10s skipped: %s\n' "$1" "$2"; }

# THE MODEL. Chosen so the optimum is UNIQUE -- a model with ties would let two correct bindings
# disagree, and this gate would blame them for it.
#
# It is the SAME model `check-semantics.sh` uses, deliberately: that gate proves every surface
# BUILDS it to identical bytes, and this one proves they SOLVE it to the same answer. Together they
# cover a program that touches two categoricals, an integer, `not_equal`, a counting constraint,
# `fix`, and a two-term objective -- rather than each gate covering a different toy.
#
#   a, b over {0,1,2}; t over 10..=13
#   a != b, at_most 1 of {a=0, b=0}, fix t = 12
#   maximise 3*[a=1] + 4*[b=2]
#
# Enumerated before being relied on: exactly one assignment scores 7, (a=1, b=2, t=12); every other
# feasible point scores 4, 3 or 0. A model with TIES would let two correct bindings disagree and
# this gate would blame them for it.
#
# `t` earns its place twice over -- it is the case where `fix(t, 12)` must mean TWELVE and not slot
# twelve, which is a confusion this project has actually shipped.
EXPECT="a=1 b=2 t=12 feasible=true"

attempted=""
attempt() { attempted="$attempted $1"; }

# ---- rust: the reference -------------------------------------------------------------------------
mkdir -p examples
cat > examples/_ans.rs <<'RS'
fn main() {
    use ferrotherm::model::{Expr, Lit, Model, Sense};
    let mut m = Model::new();
    let a = m.categorical("a", 3);
    let b = m.categorical("b", 3);
    let t = m.integer("t", 10, 13);
    m.not_equal(a, b);
    m.at_most(vec![Lit::Is(a, 0), Lit::Is(b, 0)], 1);
    m.fix(t, 12);
    m.objective(Sense::Maximize, Expr::product(3.0, &[Lit::Is(a, 1)]));
    m.objective(Sense::Maximize, Expr::product(4.0, &[Lit::Is(b, 2)]));
    let s = m.compile().unwrap().solve_best_of(64);
    print!("a={} b={} t={} feasible={}", s.value("a"), s.value("b"), s.value("t"), s.feasible());
}
RS
attempt rust
cargo run --release --quiet --example _ans > "$out/rust.txt" 2>/dev/null
rm -f examples/_ans.rs

# ---- python ---------------------------------------------------------------------------------------
attempt python
python3 - "$out" 2> "$out/python.err" <<'PY'
import sys, os
sys.path.insert(0, "python")
import ferrotherm as ft
p = ft.Problem()
a = p.categorical("a", 3); b = p.categorical("b", 3); t = p.integer("t", 10, 13)
p.not_equal(a, b)
p.at_most([a.is_(0), b.is_(0)], 1)
p.fix(t, 12)
p.maximize(3 * a.is_(1) + 4 * b.is_(2))
ans = p.solve(tries=64)
open(os.path.join(sys.argv[1], "python.txt"), "w").write(
    f"a={ans.values['a']} b={ans.values['b']} t={ans.values['t']} "
    f"feasible={str(ans.feasible).lower()}")
PY

# ---- julia ----------------------------------------------------------------------------------------
if command -v julia >/dev/null 2>&1; then
  attempt julia
  FERROTHERM_LIB="$LIB" julia --project=julia/Ferrotherm -e '
    using Ferrotherm
    p = Problem()
    a = categorical!(p, "a", 3); b = categorical!(p, "b", 3); t = integer!(p, "t", 10:13)
    not_equal!(p, a, b)
    at_most!(p, [is(a,0), is(b,0)], 1)
    fix!(p, t, 12)
    maximize!(p, [(3.0, is(a,1)), (4.0, is(b,2))])
    ans = solve!(p; tries = 64)
    print("a=", ans.values["a"], " b=", ans.values["b"], " t=", ans.values["t"],
          " feasible=", lowercase(string(feasible(ans))))
  ' > "$out/julia.txt" 2> "$out/julia.err"
else skip julia "no julia on PATH"; fi

# ---- zig ------------------------------------------------------------------------------------------
# Deliberately does NOT call compile() -- that omission is the bug this gate exists for, and a gate
# that works around the defect it hunts is worth nothing.
if command -v zig >/dev/null 2>&1; then
  attempt zig
  cat > zig/_ans.zig <<'ZG'
const std = @import("std");
const ft = @import("ferrotherm.zig");
pub fn main() !void {
    var p = try ft.Problem.init();
    defer p.deinit();
    const a = try p.categorical("a", 3);
    const b = try p.categorical("b", 3);
    const t = try p.integer("t", 10, 13);
    try p.notEqual(a, b);
    try p.count(.at_most, 1, &.{ a.is(0), b.is(0) });
    try p.fix(t, 12);
    try p.prefer(.maximize, 3.0, a.is(1));
    try p.prefer(.maximize, 4.0, b.is(2));
    try p.solve(64);
    std.debug.print("a={?d} b={?d} t={?d} feasible={}",
        .{ try p.value(a), try p.value(b), try p.value(t), p.feasible() });
}
ZG
  # Zig prints through std.debug.print, which is STDERR -- so its answer and any panic trace land
  # in the same stream. Captured together and split on exit code, because a run that failed should
  # report its first error line, not paste a stack trace into the comparison column.
  (cd zig && zig run _ans.zig -I "$here/include" -L "$here/target/release" -lferrotherm -lc \
      2> "$out/zig.raw" >/dev/null)
  if [ $? -eq 0 ]; then
    cp "$out/zig.raw" "$out/zig.txt"
  else
    head -1 "$out/zig.raw" > "$out/zig.err"
    : > "$out/zig.txt"
  fi
  rm -f zig/_ans.zig
else skip zig "no zig on PATH"; fi

# ---- http and mcp ---------------------------------------------------------------------------------
if cargo build --release --quiet -p ferrotherm-serve 2>/dev/null; then
  attempt http
  (./target/release/ferrotherm-serve >/dev/null 2>&1 &)
  for _ in $(seq 1 40); do curl -s -o /dev/null localhost:8479/v1/health 2>/dev/null && break; sleep 0.25; done
  curl -s -X POST localhost:8479/v1/solve -d '{
      "variables":[{"name":"a","values":3},{"name":"b","values":3},{"name":"t","lo":10,"hi":13}],
      "constraints":[{"type":"not_equal","a":"a","b":"b"},
                     {"type":"at_most","k":1,"of":[{"var":"a","value":0},{"var":"b","value":0}]},
                     {"type":"fix","var":"t","value":12}],
      "objective":{"maximize":true,"terms":[{"var":"a","value":1,"weight":3},{"var":"b","value":2,"weight":4}]},
      "tries":64}' 2>/dev/null \
    | python3 -c "
import json,sys
d=json.load(sys.stdin)
print(f\"a={d['values']['a']} b={d['values']['b']} t={d['values']['t']} feasible={str(d['feasible']).lower()}\", end='')" \
    > "$out/http.txt" 2> "$out/http.err"
  pkill -f 'target/release/ferrotherm-serve' 2>/dev/null
else skip http "serve did not build"; fi

# ---- compare ---------------------------------------------------------------------------------------
echo
echo "  one model with a UNIQUE optimum, solved on every surface:"
echo
bad=0; n=0
for name in $attempted; do
  f="$out/$name.txt"
  # `[ -f ]` first: a shell REDIRECT on a missing file reports through the shell's own stderr, which
  # `2>/dev/null` on the command does not catch -- so a surface that produced nothing printed a raw
  # "No such file or directory" above its own diagnosis.
  got=""
  [ -f "$f" ] && got=$(tr -d '\r\n' < "$f" 2>/dev/null)
  if [ -z "$got" ]; then
    say "$name" "PRODUCED NOTHING -- its toolchain is present, so it broke"
    [ -s "$out/$name.err" ] && sed 's/^/      /' "$out/$name.err" | tail -4
    bad=$((bad + 1)); n=$((n + 1)); continue
  fi
  n=$((n + 1))
  if [ "$got" = "$EXPECT" ]; then
    say "$name" "$got"
  else
    say "$name" "$got   <- expected $EXPECT"
    bad=$((bad + 1))
  fi
done

echo
# A floor. Without it, a machine missing every toolchain reports success having compared nothing.
if [ "$n" -lt 2 ]; then
  echo "only $n surface(s) answered; this compared nothing" >&2
  exit 2
fi
if [ "$bad" -gt 0 ]; then
  echo "$bad of $n surfaces did not return the agreed answer." >&2
  echo "Building the same model is not the same as solving it -- that gap is one function call" >&2
  echo "wide, and Zig's missing compile-before-solve lived in it." >&2
  exit 1
fi
echo "  $n surfaces solve one model to the same answer"
