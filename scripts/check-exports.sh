#!/usr/bin/env bash
#
# Does every name a binding EXPORTS actually resolve to something?
#
# `check-parity.sh` proves a C ABI symbol is declared in each binding. For Julia that means an
# `@cfn` line, which says the symbol can be called and nothing about whether anything calls it.
# Both OMMX functions were once declared and exported with no function body in between: the module
# loaded cleanly, parity passed, and `from_ommx` was an `UndefVarError` the first time anyone
# reached for it. Julia does not resolve an exported name until it is used, so nothing complained.
#
# This asks the interpreter to resolve every exported name. It is the cheapest possible check and it
# would have caught that immediately.
#
#   scripts/check-exports.sh

set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

LIB="$here/target/release/libferrotherm.dylib"
[ -f "$LIB" ] || LIB="$here/target/release/libferrotherm.so"
[ -f "$LIB" ] || { echo "build the library first: cargo build --release" >&2; exit 2; }

# A skip that CANNOT be silent where it matters.
#
# This gate lived in a CI job with no Julia installed, so it printed "skipping", exited 0, and had
# never checked anything there -- while reading, in the workflow, as a step that passed. Locally a
# skip is right (not every machine has Julia); in CI it is the failure mode this whole script exists
# to catch, one level up. `FERROTHERM_REQUIRE_JULIA=1` turns the skip into a refusal, and CI sets it.
if ! command -v julia >/dev/null 2>&1; then
  if [ "${FERROTHERM_REQUIRE_JULIA:-0}" = "1" ]; then
    echo "FERROTHERM_REQUIRE_JULIA=1 but no julia on PATH: this run would have checked nothing" >&2
    exit 2
  fi
  echo "no julia on PATH; skipping (set FERROTHERM_REQUIRE_JULIA=1 to make this a failure)"
  exit 0
fi

# Written to a file rather than passed inline: the probe contains quotes, and nesting them inside a
# command substitution is how the first version silently failed to run at all.
probe=$(mktemp); trap 'rm -f "$probe"' EXIT
cat > "$probe" <<'JL'
using Ferrotherm
ns = filter(n -> n !== :Ferrotherm, names(Ferrotherm))
miss = String[]
for n in ns
    isdefined(Ferrotherm, n) || push!(miss, String(n))
end
println(length(ns), " ", join(miss, ","))
JL
out=$(FERROTHERM_LIB="$LIB" julia --project=julia/Ferrotherm "$probe" 2>&1 | tail -1)

count=${out%% *}
missing=${out#* }
if ! [[ "$count" =~ ^[0-9]+$ ]]; then
  echo "julia could not load the module:" >&2
  echo "$out" >&2
  exit 2
fi
# A floor: an export list this short means the module did not load what it should have.
if [ "$count" -lt 40 ]; then
  echo "only $count exported names; that cannot be right" >&2
  exit 2
fi
if [ -n "$missing" ] && [ "$missing" != "$out" ]; then
  echo "  exported but undefined: $missing" >&2
  echo
  echo "Julia does not resolve an exported name until it is used, so a name exported with no" >&2
  echo "definition behind it loads cleanly and fails at the call site." >&2
  exit 1
fi
echo "  all $count exported Julia names resolve"
