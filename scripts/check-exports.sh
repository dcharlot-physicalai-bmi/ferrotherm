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
#   scripts/check-exports.sh              every exported Julia name resolves
#   scripts/check-exports.sh --selftest   prove that check can fail

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
  # A selftest that skips is the very thing a selftest is for: it would print a pass having damaged
  # nothing and checked nothing. So --selftest refuses rather than skips, whatever CI has set.
  if [ "${1:-}" = "--selftest" ]; then
    echo "SELFTEST FAILED: no julia on PATH, so nothing could be damaged and nothing was checked" >&2
    exit 2
  fi
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

# The probe, and the reading of what it says, factored into two functions -- so that the --selftest
# below runs the SAME comparison this gate runs, rather than a second copy of it free to drift away
# from the one that matters.
run_probe() {  # <project dir> -> the probe's last line of output
  FERROTHERM_LIB="$LIB" julia --project="$1" "$probe" 2>&1 | tail -1
}

verdict() {  # <probe output> -> 0 every export resolves, 1 one does not, 2 nothing was checked
  local out count missing
  out="$1"
  count=${out%% *}
  missing=${out#* }
  if ! [[ "$count" =~ ^[0-9]+$ ]]; then
    echo "julia could not load the module:" >&2
    echo "$out" >&2
    return 2
  fi
  # A floor: an export list this short means the module did not load what it should have.
  if [ "$count" -lt 40 ]; then
    echo "only $count exported names; that cannot be right" >&2
    return 2
  fi
  if [ -n "$missing" ] && [ "$missing" != "$out" ]; then
    echo "  exported but undefined: $missing" >&2
    echo >&2
    echo "Julia does not resolve an exported name until it is used, so a name exported with no" >&2
    echo "definition behind it loads cleanly and fails at the call site." >&2
    return 1
  fi
  echo "  all $count exported Julia names resolve"
  return 0
}

if [ "${1:-}" = "--selftest" ]; then
  # WHY THIS DAMAGE, and what real regression it stands in for.
  #
  # The drift this gate exists to catch is a name that is still EXPORTED after the thing behind it
  # stopped existing: a definition renamed during a refactor, or deleted while the export line
  # stayed. That is precisely how `from_ommx` shipped once -- exported, no body, module loads
  # clean, check-parity green, and an UndefVarError waiting at the first call site. So the damage
  # renames the DEFINITION of `from_ommx` and leaves the export list untouched. Nothing else in the
  # module calls it, so the copy still loads: the only thing wrong with it is the one thing this
  # gate is supposed to see, which is what makes it a fair test rather than a strawman.
  #
  # Deleting the export line instead would be the opposite mistake -- that leaves the library
  # consistent and proves nothing. Emptying the file would be caught by the `count < 40` floor or
  # by julia refusing to load, neither of which is this check working.
  #
  # It all happens in a mktemp copy of the package. This tree is shared with another session and a
  # half-restored source file is the worst outcome available, so no tracked file is touched, ever,
  # not even for an instant.
  tmp=$(mktemp -d)
  trap 'rm -f "$probe"; rm -rf "$tmp"' EXIT
  mkdir -p "$tmp/Ferrotherm/src"
  cp julia/Ferrotherm/Project.toml "$tmp/Ferrotherm/Project.toml"
  sed 's/^function from_ommx(/function from_ommx_renamed_by_selftest(/' \
    julia/Ferrotherm/src/Ferrotherm.jl > "$tmp/Ferrotherm/src/Ferrotherm.jl"

  # PROVE THE DAMAGE LANDED. A sed that matched nothing leaves an undamaged copy, and the check
  # below would then pass its own selftest by failing for some unrelated reason -- a vacuous
  # selftest wearing a green tick. So: the new name must be there, the old definition must be gone,
  # and the export must have SURVIVED, because an export that went with its definition is a
  # different and harmless edit.
  if ! grep -q '^function from_ommx_renamed_by_selftest(' "$tmp/Ferrotherm/src/Ferrotherm.jl"; then
    echo "SELFTEST FAILED: the damage did not apply -- nothing matched 'function from_ommx(' in" >&2
    echo "julia/Ferrotherm/src/Ferrotherm.jl, so the copy below is undamaged and proves nothing" >&2
    exit 1
  fi
  if grep -q '^function from_ommx(' "$tmp/Ferrotherm/src/Ferrotherm.jl"; then
    echo "SELFTEST FAILED: the damage did not apply -- a definition of from_ommx survives in the" >&2
    echo "copy, so the name still resolves and the check below is vacuous" >&2
    exit 1
  fi
  if ! grep -q '^export .*[ ,]from_ommx$' "$tmp/Ferrotherm/src/Ferrotherm.jl"; then
    echo "SELFTEST FAILED: the copy does not export from_ommx, so the damage is not the drift this" >&2
    echo "gate exists to catch -- an export left behind by its own definition" >&2
    exit 1
  fi

  damaged=$(run_probe "$tmp/Ferrotherm")
  verdict "$damaged" >/dev/null 2>&1
  rc=$?
  if [ "$rc" = 0 ]; then
    echo "SELFTEST FAILED: from_ommx is exported with nothing behind it and every exported name" >&2
    echo "still reported as resolving, so this gate would not see the drift it was written for" >&2
    exit 1
  fi
  if [ "$rc" != 1 ]; then
    echo "SELFTEST FAILED: the damaged copy did not load at all (probe said: $damaged), so the" >&2
    echo "failure proves the module can break, not that this gate can see an unresolved export" >&2
    exit 1
  fi
  # And for the RIGHT name. A failure naming some other export would mean the copy is broken in a
  # way that has nothing to do with the damage, and the pass above would be luck.
  if ! printf '%s' "$damaged" | grep -q 'from_ommx'; then
    echo "SELFTEST FAILED: the check failed, but it did not name from_ommx (probe said: $damaged)," >&2
    echo "so something other than the injected damage was caught" >&2
    exit 1
  fi
  echo "selftest: an export whose definition was renamed away was caught ($damaged), so this can fail"
  exit 0
fi

out=$(run_probe "julia/Ferrotherm")
verdict "$out"
exit $?
