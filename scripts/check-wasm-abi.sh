#!/usr/bin/env bash
#
# Does the committed wasm have the ABI the source has?
#
# check-wasm-exports.sh asks whether every symbol the pages call is EXPORTED. It cannot see a symbol
# whose signature changed. That is how docs/ferrotherm.wasm went stale for three weeks: `ft_ebm_train`
# gained a `method` parameter on 2026-09-09 (af6d066), the C header and the Python, Julia and Zig
# bindings were updated, and the committed wasm -- last rebuilt on 2026-09-08 -- kept the old ten
# parameters. Two pages and scripts/fit-wasm.mjs kept calling the old ten, which matched the stale
# binary, so check-fit.sh agreed with itself and every gate stayed green. When the wasm was finally
# rebuilt (2026-09-28) the calls broke: a BigInt seed landed in a u32 slot.
#
# Bytes are not comparable across toolchains (see check-wasm-exports.sh), but SIGNATURES are: the
# parameter and result types of an `extern "C"` function are fixed by the Rust source, not by the
# compiler version. So this compares, export by export, the committed binary against a fresh build
# from the source being tested, and fails on any difference -- a changed signature, an export the
# source has and the committed binary lacks, or one the committed binary still carries after the
# source removed it.
#
#   scripts/check-wasm-abi.sh <fresh.wasm> [committed.wasm]   default committed: docs/ferrotherm.wasm
#   scripts/check-wasm-abi.sh --selftest                        prove the comparison can fail
#
# CI builds the fresh binary in the same job:
#   RUSTFLAGS='-C strip=symbols' cargo build --release --lib --target wasm32-unknown-unknown

set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

compare() {
  python3 - "$1" "$2" <<'PY'
import sys

VT = {0x7F: "i32", 0x7E: "i64", 0x7D: "f32", 0x7C: "f64", 0x7B: "v128", 0x70: "funcref", 0x6F: "externref"}

def uleb(b, i):
    v = s = 0
    while True:
        c = b[i]; i += 1
        v |= (c & 0x7F) << s
        if not c & 0x80:
            return v, i
        s += 7

def signatures(path):
    b = open(path, "rb").read()
    if b[:4] != b"\0asm":
        print("%s is not a wasm binary" % path, file=sys.stderr); sys.exit(2)
    i, types, funcs, imported, exports = 8, [], [], 0, {}
    while i < len(b):
        sid = b[i]; i += 1
        size, i = uleb(b, i)
        end = i + size
        if sid == 1:                                   # types
            n, j = uleb(b, i)
            for _ in range(n):
                j += 1                                 # 0x60, func
                np_, j = uleb(b, j); ps = tuple(VT.get(b[j + k], hex(b[j + k])) for k in range(np_)); j += np_
                nr, j = uleb(b, j); rs = tuple(VT.get(b[j + k], hex(b[j + k])) for k in range(nr)); j += nr
                types.append((ps, rs))
        elif sid == 2:                                 # imports: only function imports shift indices
            n, j = uleb(b, i)
            for _ in range(n):
                l, j = uleb(b, j); j += l
                l, j = uleb(b, j); j += l
                kind = b[j]; j += 1
                if kind == 0:
                    _, j = uleb(b, j); imported += 1
                elif kind == 1:
                    j += 1; flags, j = uleb(b, j); _, j = uleb(b, j)
                    if flags & 1: _, j = uleb(b, j)
                elif kind == 2:
                    flags, j = uleb(b, j); _, j = uleb(b, j)
                    if flags & 1: _, j = uleb(b, j)
                else:
                    j += 2
        elif sid == 3:                                 # functions: type index per defined function
            n, j = uleb(b, i)
            for _ in range(n):
                t, j = uleb(b, j); funcs.append(t)
        elif sid == 7:                                 # exports
            n, j = uleb(b, i)
            for _ in range(n):
                l, j = uleb(b, j); name = b[j:j + l].decode(); j += l
                kind = b[j]; j += 1
                idx, j = uleb(b, j)
                if kind == 0 and idx >= imported:
                    exports[name] = types[funcs[idx - imported]]
        i = end
    return exports

def show(sig):
    return "(%s) -> (%s)" % (", ".join(sig[0]), ", ".join(sig[1]))

fresh, committed = signatures(sys.argv[1]), signatures(sys.argv[2])
bad = []
for name in sorted(set(fresh) | set(committed)):
    if name not in committed:
        bad.append("%s: in the source, missing from the committed wasm" % name)
    elif name not in fresh:
        bad.append("%s: in the committed wasm, no longer in the source" % name)
    elif fresh[name] != committed[name]:
        bad.append("%s: committed %s, source %s" % (name, show(committed[name]), show(fresh[name])))
if bad:
    print("the committed wasm's ABI differs from the source in %d export(s):" % len(bad), file=sys.stderr)
    for line in bad:
        print("  " + line, file=sys.stderr)
    sys.exit(1)
print("committed wasm matches the source's ABI: %d exported functions, every signature equal" % len(fresh))
PY
}

if [[ "${1:-}" == "--selftest" ]]; then
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT
  # Two minimal modules, one exported function each, identical but for one parameter: what
  # ft_ebm_train looked like before and after it gained `method`.
  python3 - "$tmp" <<'PY'
import sys
d = sys.argv[1]
def uleb(n):
    out = bytearray()
    while True:
        b = n & 0x7F; n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)
def section(sid, payload):
    return bytes([sid]) + uleb(len(payload)) + payload
def module(params):
    ty = uleb(1) + b"\x60" + uleb(len(params)) + bytes(params) + uleb(1) + b"\x7f"
    fn = uleb(1) + uleb(0)
    name = b"ft_ebm_train"
    ex = uleb(1) + uleb(len(name)) + name + b"\x00" + uleb(0)
    body = uleb(0) + b"\x41\x00\x0b"                  # no locals; i32.const 0; end
    code = uleb(1) + uleb(len(body)) + body
    return b"\0asm\x01\0\0\0" + section(1, ty) + section(3, fn) + section(7, ex) + section(10, code)
old = [0x7F] * 7 + [0x7C, 0x7F, 0x7E]                 # ten parameters
new = [0x7F] * 8 + [0x7C, 0x7F, 0x7E]                 # eleven: `method` added
open(d + "/old.wasm", "wb").write(module(old))
open(d + "/new.wasm", "wb").write(module(new))
PY
  # Positive controls: a module against itself, and the real committed binary against itself.
  if ! compare "$tmp/new.wasm" "$tmp/new.wasm" >/dev/null 2>&1; then
    echo "SELFTEST FAILED: a module does not match itself, so no failure below is attributable" >&2
    exit 1
  fi
  if ! compare "$here/docs/ferrotherm.wasm" "$here/docs/ferrotherm.wasm" >/dev/null 2>"$tmp/self.err"; then
    echo "SELFTEST FAILED: the committed wasm does not match itself:" >&2
    sed 's/^/  /' "$tmp/self.err" >&2
    exit 1
  fi
  # The damage: a committed binary one parameter behind the source.
  if compare "$tmp/new.wasm" "$tmp/old.wasm" >/dev/null 2>"$tmp/stale.err"; then
    echo "SELFTEST FAILED: a committed wasm missing a parameter passed" >&2
    exit 1
  fi
  if ! grep -q 'ft_ebm_train: committed (i32, i32, i32, i32, i32, i32, i32, f64, i32, i64)' "$tmp/stale.err"; then
    echo "SELFTEST FAILED: the gate failed, but did not name the stale signature -- it said:" >&2
    sed 's/^/  /' "$tmp/stale.err" >&2
    exit 1
  fi
  echo "selftest: a committed wasm one parameter behind its source was caught, and named"
  exit 0
fi

fresh="${1:?usage: scripts/check-wasm-abi.sh <fresh.wasm> [committed.wasm]}"
committed="${2:-$here/docs/ferrotherm.wasm}"
for f in "$fresh" "$committed"; do
  [[ -f "$f" ]] || { echo "no wasm at $f" >&2; exit 2; }
done
if ! compare "$fresh" "$committed"; then
  echo >&2
  echo "Rebuild the committed binary from this source, then fix every page call whose signature moved:" >&2
  echo "  RUSTFLAGS='-C strip=symbols' cargo build --release --lib --target wasm32-unknown-unknown" >&2
  echo "  cp target/wasm32-unknown-unknown/release/ferrotherm.wasm docs/" >&2
  exit 1
fi
