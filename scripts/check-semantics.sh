#!/usr/bin/env bash
#
# Do the bindings COMPUTE the same thing, or merely expose the same names?
#
# `check-parity.sh` proves every C ABI symbol reaches every binding. That is a real check and it is
# not this one: nine surfaces can each have `all_different` and three of them can get it wrong.
# Symbol parity is not semantic parity.
#
# The `.ftp` program is the semantic fingerprint -- spins, biases, factors, colour classes,
# encodings, schedule -- so two surfaces that build the same model must emit the same bytes.
# Anything else means a binding is building a different model than it looks like it is building,
# which is the class of bug this project has shipped repeatedly: a literal carrying a slot index
# where every name said value, a `maximize` that minimised, an `encoding` field read by nobody.
#
# It found one on its first run. Rust and Python disagreed -- same length, same factors, different
# ORDER -- which turned out not to be a binding bug but `GraphBuilder::build` merging through a
# HashMap, whose iteration order Rust randomises per instance. Five runs of one binding gave five
# different programs. Fixed in 0.10.0+; this is what keeps it fixed.
#
# A missing toolchain SKIPS rather than fails -- a Linux runner has no Zig or Julia, and a red suite
# that means "this machine lacks a compiler" is a suite people stop reading. The floor below is what
# stops that becoming a check that passes over nothing.
#
#   scripts/check-semantics.sh

set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
out=$(mktemp -d)
cleanup() { rm -rf "$out"; pkill -f 'target/release/ferrotherm-serve' 2>/dev/null; }
trap cleanup EXIT

# Build the cdylib the bindings dlopen. NOTHING DID.
#
# `cargo run --example` below builds the library as an rlib for the example to link; it does not
# emit the cdylib. On a developer machine one is always lying in target/release from some earlier
# `cargo build`, so this passed everywhere it was run by hand -- and on a clean runner python and
# julia, the two bindings that load the shared library, could never find it. CI was red for weeks
# with "PRODUCED NOTHING" and the cause was that this script never built what it needs.
cargo build --release --quiet --lib 2>/dev/null || {
  echo "the library itself did not build" >&2; exit 2; }

LIB="$here/target/release/libferrotherm.dylib"
[ -f "$LIB" ] || LIB="$here/target/release/libferrotherm.so"
[ -f "$LIB" ] || { echo "no cdylib at target/release after building it" >&2; exit 2; }
say() { printf '  %-14s %s\n' "$1" "$2"; }
skip() { printf '  %-14s skipped: %s\n' "$1" "$2"; }

# Which bindings were ATTEMPTED, so one that crashes is reported rather than dropped.
#
# Without this the loop below only saw files that exist. A binding that failed outright produced no
# file, vanished from the comparison, and the run reported "5 bindings identical" -- which is true
# and useless. Found by mutating the Python binding to carry a slot index instead of a value: the
# binding raised, wrote nothing, and the check said everything agreed.
#
# A toolchain that is not installed is a SKIP; a toolchain that is installed and produced nothing is
# a FAILURE, because that means the binding broke.
attempted=""
attempt() { attempted="$attempted $1"; }

# One model, built the same way on every surface. It deliberately exercises what has been wrong
# before: an integer over 10..=13 so `fix(t, 12)` must mean TWELVE and not slot twelve, a counting
# constraint that costs a slack variable, and objective terms with different weights AND values.
cat > "$out/model.json" <<'JS'
{"variables":[{"name":"a","values":3},{"name":"b","values":3},{"name":"t","lo":10,"hi":13}],
 "constraints":[{"type":"not_equal","a":"a","b":"b"},
                {"type":"at_most","k":1,"of":[{"var":"a","value":0},{"var":"b","value":0}]},
                {"type":"fix","var":"t","value":12}],
 "objective":{"maximize":true,"terms":[{"var":"a","value":1,"weight":3},{"var":"b","value":2,"weight":4}]},
 "tries":1}
JS

# ---- rust: the reference -----------------------------------------------------------------------
mkdir -p examples
cat > examples/_sem.rs <<'RS'
fn main() {
    use ferrotherm::encode::Encoding;
    use ferrotherm::model::{Expr, Lit, Model, Sense};
    let mut m = Model::new();
    let a = m.categorical_as("a", 3, Encoding::OneHot);
    let b = m.categorical_as("b", 3, Encoding::OneHot);
    let t = m.integer_as("t", 10, 13, Encoding::OneHot);
    m.not_equal(a, b);
    m.at_most(vec![Lit::Is(a, 0), Lit::Is(b, 0)], 1);
    m.fix(t, 12);
    m.objective(Sense::Maximize, Expr::product(3.0, &[Lit::Is(a, 1)]));
    m.objective(Sense::Maximize, Expr::product(4.0, &[Lit::Is(b, 2)]));
    print!("{}", m.compile().unwrap().program.to_ftp());
}
RS
cargo run --release --quiet --example _sem > "$out/rust.ftp" 2>/dev/null
rm -f examples/_sem.rs
[ -s "$out/rust.ftp" ] || { echo "the reference itself did not build" >&2; exit 2; }
say "rust" "$(shasum -a 256 < "$out/rust.ftp" | cut -c1-16)  reference, $(wc -c < "$out/rust.ftp" | tr -d ' ') bytes"

# ---- python ------------------------------------------------------------------------------------
attempt python
python3 - "$out" 2> "$out/python.err" <<'PY'
import sys, os
sys.path.insert(0, "python")
import ferrotherm as ft
p = ft.Problem()
a = p.categorical("a", 3); b = p.categorical("b", 3); t = p.integer("t", 10, 13)
p.not_equal(a, b); p.at_most([a.is_(0), b.is_(0)], 1); p.fix(t, 12)
p.maximize(3 * a.is_(1) + 4 * b.is_(2)); p.solve(tries=1)
open(os.path.join(sys.argv[1], "python.ftp"), "w").write(p.ftp())
PY

# ---- julia -------------------------------------------------------------------------------------
if command -v julia >/dev/null 2>&1; then
  attempt julia
  FERROTHERM_LIB="$LIB" julia --project=julia/Ferrotherm -e '
    using Ferrotherm
    p = Problem()
    a = categorical!(p, "a", 3); b = categorical!(p, "b", 3); t = integer!(p, "t", 10:13)
    not_equal!(p, a, b); at_most!(p, [is(a,0), is(b,0)], 1); fix!(p, t, 12)
    maximize!(p, [(3.0, is(a,1)), (4.0, is(b,2))]); solve!(p; tries = 1)
    print(ftp(p))' > "$out/julia.ftp" 2> "$out/julia.err"
else skip julia "no julia on PATH"; fi

# ---- zig ---------------------------------------------------------------------------------------
# Writes to STDERR via std.debug.print: the stdout and file APIs moved between 0.14 and 0.16 and are
# not worth chasing for a one-model dump, while debug.print has been stable throughout.
if command -v zig >/dev/null 2>&1; then
  attempt zig
  cat > zig/_sem.zig <<'ZG'
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
    _ = try p.compile();
    var buf: [65536]u8 = undefined;
    std.debug.print("{s}", .{p.ftp(&buf)});
}
ZG
  (cd zig && zig run _sem.zig -I "$here/include" -L "$here/target/release" -lferrotherm -lc \
     2> "$out/zig.ftp" >/dev/null)
  rm -f zig/_sem.zig
else skip zig "no zig on PATH"; fi

# ---- http and mcp: the real binaries, not a stand-in -------------------------------------------
# A crate in THIS repository failing to compile is a failure, not an absent toolchain.
#
# This was `cargo build ... 2>/dev/null` falling through to `skip http/mcp "serve did not build"`,
# which reads like "this machine lacks something" and exits 0. It does not lack anything: the crate
# is right here and it is broken. The compiler error was discarded on the way past.
if ! cargo build --release --quiet -p ferrotherm-serve 2> "$out/serve.err"; then
  echo "  ferrotherm-serve is in this repository and did not compile:" >&2
  sed 's/^/      /' "$out/serve.err" | tail -20 >&2
  echo "  That is a break, not a missing toolchain." >&2
  exit 2
fi
if true; then
  attempt http; attempt mcp
  (./target/release/ferrotherm-serve >/dev/null 2>&1 &)
  for _ in $(seq 1 40); do curl -s -o /dev/null localhost:8479/v1/health 2>/dev/null && break; sleep 0.25; done
  curl -s -X POST localhost:8479/v1/solve -d @"$out/model.json" 2>/dev/null \
    | python3 -c "import json,sys;print(json.load(sys.stdin)['ftp'],end='')" > "$out/http.ftp" 2> "$out/http.err"
  python3 - "$out" 2> "$out/mcp.err" <<'PY'
import json, subprocess, sys, os
args = json.load(open(os.path.join(sys.argv[1], "model.json")))
reqs = [{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"parity","version":"1"}}},
        {"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"ferrotherm_solve","arguments":args}}]
p = subprocess.run(["./target/release/ferrotherm-mcp"], input="".join(json.dumps(r)+"\n" for r in reqs),
                   capture_output=True, text=True, timeout=120)
for line in p.stdout.splitlines():
    try: m = json.loads(line)
    except Exception: continue
    if m.get("id") == 2:
        open(os.path.join(sys.argv[1], "mcp.ftp"), "w").write(json.loads(m["result"]["content"][0]["text"])["ftp"])
        break
PY
  pkill -f 'target/release/ferrotherm-serve' 2>/dev/null
fi

# ---- the editor, which is the wasm ---------------------------------------------------------------
if [ -f web-tests/node_modules/.package-lock.json ] || node -e "require('playwright')" 2>/dev/null; then
  attempt wasm
  cp "$out/model.json" /tmp/_sem_model.json
  node scripts/sem-wasm.mjs > "$out/wasm.ftp" 2> "$out/wasm.err" || true
else skip wasm "no playwright"; fi

# ---- compare -------------------------------------------------------------------------------------
echo
bad=0; n=0
for name in $attempted; do
  f="$out/$name.ftp"
  if [ ! -s "$f" ]; then
    say "$name" "PRODUCED NOTHING -- its toolchain is present, so it broke"
    # And say HOW. This printed the verdict and swallowed the cause on every binding, so weeks of
    # red CI showed "it broke" while the discarded stderr read "could not load the ferrotherm
    # shared library. Build it with cargo build --release" -- the fix, in the message, thrown away.
    if [ -s "$out/$name.err" ]; then
      sed 's/^/      /' "$out/$name.err" | tail -6
    else
      echo "      (it wrote nothing to stderr either)"
    fi
    bad=$((bad + 1)); n=$((n + 1)); continue
  fi
  n=$((n + 1))
  if cmp -s "$f" "$out/rust.ftp"; then
    say "$name" "$(shasum -a 256 < "$f" | cut -c1-16)  identical"
  else
    say "$name" "$(shasum -a 256 < "$f" | cut -c1-16)  DIFFERS"
    diff <(sort "$out/rust.ftp") <(sort "$f") | head -8 | sed 's/^/      /'
    bad=$((bad + 1))
  fi
done

echo
# A floor. Without it, a machine missing every toolchain reports success having compared nothing --
# which is the shape of a green check that means "nothing ran".
if [ "$n" -lt 2 ]; then
  echo "only $n binding(s) produced a program; this compared nothing" >&2
  exit 2
fi
if [ "$bad" -gt 0 ]; then
  echo "$bad of $n bindings compile one model to different bytes." >&2
  echo "A binding that emits a different program is building a different model." >&2
  exit 1
fi
echo "  $n bindings compile one model to identical bytes"
