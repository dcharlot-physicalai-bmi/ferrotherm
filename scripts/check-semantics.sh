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
#   scripts/check-semantics.sh              compile one model through every binding, compare bytes
#   scripts/check-semantics.sh --selftest   prove the comparison can fail
#
# --selftest adds three EXTRA arms to the comparison, each built by a copy of the python package
# with one field quietly dropped, and demands that all three are caught while every honest binding
# still agrees. It exists because this gate's whole output is one hash repeated seven times: a
# comparison that had stopped comparing -- a reference read from the wrong file, a `cmp` on two
# empty files, a loop over an empty list -- prints exactly the same green line as a healthy tree.

set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
SELFTEST=0
if [ "${1:-}" = "--selftest" ]; then SELFTEST=1; fi
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
note_skip() { missing="$missing $1"; }
skip() { note_skip "$1"; printf '  %-14s skipped: %s\n' "$1" "$2"; }


# A skip that CANNOT be silent where it matters.
#
# These gates compare SURFACES, so a missing toolchain does not weaken them a little -- it removes a
# whole surface from the comparison and still exits 0. Measured: in CI's `test` job, which had only
# Rust, this compared four of seven and reported success. That is the same shape as
# `check-exports.sh` sitting in a job with no Julia.
#
# Locally a skip is right; not every machine has Julia, Zig and a browser. In CI it is the failure
# mode the gate exists to catch, one level up. `FERROTHERM_REQUIRE_ALL=1` turns every skip into a
# refusal, and CI sets it.
require_all() { [ "${FERROTHERM_REQUIRE_ALL:-0}" = "1" ]; }
missing=""
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
#
# Written to a FILE rather than fed on stdin because --selftest runs this same program against a
# damaged copy of the package. A second, pasted copy of the model builder could drift from this one
# -- and a selftest that builds a different model than the gate compares is proving nothing about
# the gate. argv is (output directory, directory to import ferrotherm from, name of this arm);
# passing "python" for the second reproduces the old `sys.path.insert(0, "python")` exactly.
cat > "$out/build.py" <<'PY'
import sys, os
sys.path.insert(0, sys.argv[2])
import ferrotherm as ft
p = ft.Problem()
a = p.categorical("a", 3); b = p.categorical("b", 3); t = p.integer("t", 10, 13)
p.not_equal(a, b); p.at_most([a.is_(0), b.is_(0)], 1); p.fix(t, 12)
p.maximize(3 * a.is_(1) + 4 * b.is_(2)); p.solve(tries=1)
open(os.path.join(sys.argv[1], sys.argv[3] + ".ftp"), "w").write(p.ftp())
PY
attempt python
python3 "$out/build.py" "$out" python python 2> "$out/python.err"

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
# ONE probe, run by the same file that will do the work.
#
# This guard has now been wrong twice. It first tested for `web-tests/node_modules/.package-lock.json`
# and ran the driver from the repo root, where the import does not look. It then ran the driver from
# `web-tests`, which changes nothing: bare-specifier resolution follows the IMPORTING FILE, never the
# working directory. Both versions passed locally because a stray root `node_modules` answered the
# import in every arrangement, so the local check could not tell the cases apart.
#
# The driver resolves playwright explicitly now and `--probe` asks it whether it can. Nothing here
# re-implements that decision, because a guard and the code it guards must not be able to disagree.
if node "$here/scripts/sem-wasm.mjs" --probe 2>/dev/null; then
  attempt wasm
  cp "$out/model.json" /tmp/_sem_model.json
  node "$here/scripts/sem-wasm.mjs" > "$out/wasm.ftp" 2> "$out/wasm.err" || true
else skip wasm "playwright resolves from neither the repo root nor web-tests/"; fi

# ---- the damaged arms, only under --selftest ---------------------------------------------------
#
# WHY THESE THREE DAMAGES, and not something louder.
#
# The bug this gate exists to catch is a binding that still WORKS and means something else. Every
# example in the header is that shape: a literal carrying a slot index where every name said value,
# a `maximize` that minimised, an `encoding` field read by nobody. None of them raise, none of them
# fail to compile, and none of them are visible in a binding's own tests -- which is exactly why
# symbol parity and a green test suite both pass straight over them. So the damage here is three
# dropped fields, one per lowering path the model exercises:
#
#   k         the counting constraint closes with a constant 0 instead of the caller's k, the shape
#             of a binding that forwards a default where it meant to forward an argument. The
#             program is still valid; it is `at most none of these` where the caller said one.
#   weight    every objective term is priced 1.0 instead of its coefficient. This is the field a
#             binding drops most easily, because a wrong weight still solves and still returns an
#             answer that looks like an answer -- it is just optimising a different problem.
#   encoding  the integer is declared domain-wall while the caller asked for the default one-hot,
#             the shape of a binding whose own default has drifted from the library's. Spin count
#             and layout change; nothing errors.
#
# Each is applied to a COPY of the python package under $out, never to python/ in the tree, and
# each becomes an extra arm that goes through the same comparison loop and the same `cmp` as every
# real binding. FERROTHERM_LIB pins the copy to the very dylib the honest arms loaded, so a
# difference can only come from the damaged source and never from a second library.
#
# Deliberately NOT chosen: deleting a method (the binding raises and writes nothing, which this
# gate already reports as PRODUCED NOTHING), or emptying the file. Those prove only that a broken
# binding is noticed. The verdict below refuses to accept a damaged arm that produced no program,
# for that reason.
damaged=""
if [ "$SELFTEST" = 1 ]; then
  while IFS='|' read -r dname dsed dmark; do
    [ -n "$dname" ] || continue
    pkg="$out/damaged/$dname"
    mkdir -p "$pkg"
    cp -R python/ferrotherm "$pkg/ferrotherm" || {
      echo "SELFTEST FAILED: could not copy the python package to damage it" >&2; exit 1; }
    rm -rf "$pkg/ferrotherm/__pycache__"
    sed "$dsed" python/ferrotherm/__init__.py > "$pkg/ferrotherm/__init__.py"
    # PROVE THE DAMAGE APPLIED. A sed whose pattern no longer matches leaves a pristine copy, the
    # arm then agrees with the reference, and the selftest reports a failure that is really its own
    # patch rotting. Both halves are checked: the injected text is there, and the file actually
    # changed.
    if ! grep -qF "$dmark" "$pkg/ferrotherm/__init__.py" ||
       diff -q python/ferrotherm/__init__.py "$pkg/ferrotherm/__init__.py" >/dev/null; then
      echo "SELFTEST FAILED: the damage did not apply. Damage '$dname' expected to find" >&2
      echo "      $dmark" >&2
      echo "  in the damaged copy, and python/ferrotherm/__init__.py no longer looks the way this" >&2
      echo "  patch expects. Nothing was tested; fix the sed, not the gate." >&2
      exit 1
    fi
    arm="damage-$dname"
    attempt "$arm"
    damaged="$damaged $arm"
    FERROTHERM_LIB="$LIB" python3 "$out/build.py" "$out" "$pkg" "$arm" 2> "$out/$arm.err"
  done <<'DAMAGE'
k|s/_model_close(self._h, kind, int(k))/_model_close(self._h, kind, 0)/|_model_close(self._h, kind, 0)
weight|s/_model_objective_term(self._h, m, coeff, /_model_objective_term(self._h, m, 1.0, /|_model_objective_term(self._h, m, 1.0,
encoding|s/int(hi), _encoding(encoding)/int(hi), _encoding("domain-wall")/|_encoding("domain-wall")
DAMAGE
fi

# ---- compare -------------------------------------------------------------------------------------
echo
bad=0; n=0; differed=""
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
    bad=$((bad + 1)); differed="$differed $name"
  fi
done

echo
# The selftest verdict, which INVERTS the usual one: the damaged arms must have been caught.
#
# It reads the same `differed` the comparison loop above fills in, so neutering that comparison --
# making `cmp` always succeed, or pointing the reference at the wrong file -- turns this green
# selftest red. That is the property being bought here, and it is worth checking by hand after any
# edit to the loop: temporarily make the comparison always agree, and this must fail.
if [ "$SELFTEST" = 1 ]; then
  fail=0
  for arm in $damaged; do
    if [ ! -s "$out/$arm.ftp" ]; then
      echo "SELFTEST FAILED: $arm produced no program at all, so this run shows only that a binding" >&2
      echo "  which CRASHES is noticed -- never that one which silently means something else is." >&2
      echo "  The damage has to leave a binding that still compiles. It said:" >&2
      [ -s "$out/$arm.err" ] && sed 's/^/      /' "$out/$arm.err" | tail -6 >&2
      fail=1; continue
    fi
    case " $differed " in
      *" $arm "*) ;;
      *) echo "SELFTEST FAILED: $arm compiled to the SAME bytes as the reference." >&2
         echo "  A binding with that field dropped slipped through this comparison, so a green run" >&2
         echo "  of this gate is not evidence that the surfaces agree." >&2
         fail=1 ;;
    esac
  done
  for arm in $differed; do
    case " $damaged " in
      *" $arm "*) ;;
      *) echo "SELFTEST FAILED: $arm is undamaged and differs from the reference anyway." >&2
         echo "  That is a real semantic break in this tree, not a selftest result. Run the gate" >&2
         echo "  with no arguments." >&2
         fail=1 ;;
    esac
  done
  [ "$fail" = 0 ] || exit 1
  echo "  selftest: a dropped k, a dropped objective weight and a dropped encoding each still"
  echo "  compiled, each emitted a different program, and each was caught; every honest binding"
  echo "  agreed. This comparison can fail."
  exit 0
fi

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
if require_all && [ -n "$missing" ]; then
  echo "FERROTHERM_REQUIRE_ALL=1 and these surfaces were skipped:${missing}" >&2
  echo "A skipped surface is not a weaker comparison, it is one fewer thing compared -- and this" >&2
  echo "run would otherwise have exited 0 having said nothing about them." >&2
  exit 2
fi
echo "  $n bindings compile one model to identical bytes"
