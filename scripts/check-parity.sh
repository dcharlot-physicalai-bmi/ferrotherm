#!/usr/bin/env bash
#
# Does every C ABI symbol reach every binding that wraps it?
#
# AGENTS.md says a new capability lands on every surface it belongs on, or the gap is written down.
# That rule has been broken twice, both times the same way: a capability shipped in Rust, the other
# surfaces compiled and their tests stayed green, and nobody noticed for weeks -- because a binding
# that is MISSING a function is not a build error anywhere, it is simply a thing users cannot say.
#
# So the rule is checked rather than remembered. The C ABI in src/ffi.rs is the spine every non-Rust
# surface hangs off, and this asks, for each exported symbol, whether the header declares it and
# each binding calls it.
#
# A gap is not always a bug -- some symbols genuinely belong to one surface. Those go in the EXEMPT
# table below WITH A REASON, which is what "the gap is written down" means. An undocumented gap
# fails; a documented one prints and passes. Adding a line to EXEMPT is a deliberate act, and the
# reason is the part that matters.
#
#   scripts/check-parity.sh

set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

# symbol|surfaces it is exempt from|why
#
# Surfaces: header python zig julia. Use "all" for a symbol no binding should wrap.
EXEMPT=$(cat <<'TABLE'
ft_free|all|Frees a string this library allocated. Every binding copies into its own memory before returning, so a caller never holds one of these; exposing it would be handing out a way to double-free.
ft_scratch|header python zig julia|A wasm-only bump allocator for passing strings into the module. Native callers pass their own pointers, so there is nothing for it to do off the web.
ft_model_cardinality|zig julia|A fixed-arity form taking up to four variables positionally, for the node graph: a browser has no allocator to build a variadic call across this boundary. Zig and Julia use ft_model_lit + ft_model_close, which is the same constraint with no arity ceiling.
ft_model_at_most|zig julia|Same fixed-arity node-graph form as ft_model_cardinality; superseded by ft_model_lit + ft_model_close.
ft_model_at_least|zig julia|Same fixed-arity node-graph form as ft_model_cardinality; superseded by ft_model_lit + ft_model_close.
ft_model_lits|python zig julia|Reports how many literals are pending so a caller can check its own bookkeeping. The bindings build the list and call close in one function, so there is no window in which they could have lost count.
ft_model_categorical|zig|Zig calls ft_model_categorical_as, which takes the encoding as well. Binding both would offer a second way to say strictly less.
ft_model_integer|zig|Zig calls ft_model_integer_as, which takes the encoding as well. Binding both would offer a second way to say strictly less.
ft_model_solve|python|Python calls ft_model_solve_with, which is the same call plus the annealing ladder. Passing zero for each ladder parameter reproduces this one exactly.
ft_nodes|python zig julia|An alias for ft_len, provided in the C ABI because callers reach for both names. The bindings expose one name for one number.
ft_shader|python zig julia|The WGSL sweep source, so a browser runs the arithmetic this library tests rather than its own copy. A native caller already has the compiled kernels; handing it shader text would be handing it a second implementation to keep in step.
ft_shader_len|python zig julia|Length of ft_shader; same reasoning.
ft_field|python zig julia|Local field at one node, for checking a state computed elsewhere against this library's arithmetic node by node. The browser GPU path is the caller; the higher-level bindings expose a model, not a per-node debugger, and the header declares it for anyone who wants one.
ft_gpu_k|python zig julia|Part of the device-buffer view of the graph, so a caller building GPU buffers takes the layout from here rather than keeping a second copy. The browser is the caller; a native binding hands over the whole simulation instead. Declared in the header.
ft_gpu_nbr|python zig julia|Part of the device-buffer view of the graph; see ft_gpu_k.
ft_gpu_w|python zig julia|Part of the device-buffer view of the graph; see ft_gpu_k.
ft_gpu_h|python zig julia|Part of the device-buffer view of the graph; see ft_gpu_k.
ft_gpu_classes|python zig julia|Part of the device-buffer view of the graph; see ft_gpu_k.
ft_gpu_class_ptr|python zig julia|Part of the device-buffer view of the graph; see ft_gpu_k.
ft_gpu_class_len|python zig julia|Part of the device-buffer view of the graph; see ft_gpu_k.
ft_cert_passed|zig|Zig's Certificate.passed() reads its own findings count, which is the same number by the same rule. Calling across the boundary for a comparison it already holds would let the two answers disagree.
TABLE
)

exempt_reason() {  # symbol surface -> reason on stdout, empty if not exempt
  while IFS='|' read -r sym surfaces why; do
    [[ "$sym" == "$1" ]] || continue
    [[ " $surfaces " == *" all "* || " $surfaces " == *" $2 "* ]] && { printf '%s' "$why"; return; }
  done <<< "$EXEMPT"
}

# Every exported symbol, from the one place that defines them.
symbols=()
while IFS= read -r s; do
  [[ -n "$s" ]] && symbols+=("$s")
done < <(grep -oE 'pub extern "C" fn (ft_[a-z0-9_]+)' src/ffi.rs | awk '{print $NF}' | sort -u)

if [[ ${#symbols[@]} -lt 50 ]]; then
  # A floor, not a formality: if the grep above stops matching -- a rustfmt change, an attribute
  # macro -- this check would pass vacuously over an empty list and report perfect parity.
  echo "found only ${#symbols[@]} exported symbols in src/ffi.rs, which cannot be right" >&2
  exit 2
fi

declare -a SURFACES=(header python zig julia)
file_for() {
  case "$1" in
    header) echo "include/ferrotherm.h" ;;
    python) echo "python/ferrotherm/__init__.py" ;;
    zig)    echo "zig/ferrotherm.zig" ;;
    julia)  echo "julia/Ferrotherm/src/Ferrotherm.jl" ;;
  esac
}

# How each surface NAMES a symbol it has bound. Matching the bare name would count a mention in a
# comment or a doc string as coverage, which is the failure this check exists to catch.
pattern_for() {
  case "$1" in
    header) echo "\\b$2[[:space:]]*\\(" ;;
    python) echo "_sig\\(\"$2\"" ;;
    zig)    echo "c\\.$2\\b" ;;
    julia)  echo "^@cfn $2 " ;;
  esac
}

# An exemption naming a symbol that does not exist is a lie in the table, and a table of reasons
# nobody can check is the thing this script replaced. I wrote one on the first pass -- an entry for
# ft_abi_version, a function that has never existed here.
stale=0
while IFS='|' read -r sym _ _; do
  [[ -n "$sym" ]] || continue
  if ! printf '%s\n' "${symbols[@]}" | grep -qx "$sym"; then
    printf '  STALE   %-34s exempted, but src/ffi.rs exports no such symbol\n' "$sym"
    stale=$((stale + 1))
  fi
done <<< "$EXEMPT"

missing=0
exempted=0
for sym in "${symbols[@]}"; do
  for surface in "${SURFACES[@]}"; do
    f="$(file_for "$surface")"
    if grep -qE "$(pattern_for "$surface" "$sym")" "$f"; then
      continue
    fi
    why="$(exempt_reason "$sym" "$surface")"
    if [[ -n "$why" ]]; then
      printf '  exempt  %-34s %-7s %s\n' "$sym" "$surface" "$why"
      exempted=$((exempted + 1))
      continue
    fi
    printf '  MISSING %-34s %-7s not reachable from %s\n' "$sym" "$surface" "$f"
    missing=$((missing + 1))
  done
done

# ---- and the gap this check could not see -------------------------------------------------------
#
# Everything above asks whether the core's C ABI reaches the bindings. It has nothing to say about a
# capability that never entered the core at all -- which is how the OMMX bridge shipped as a sibling
# crate, reachable from Rust and from none of the other eight surfaces, past a check written
# precisely to stop that.
#
# A sibling exists because the core cannot hold it. Every one of them can say why:
#
#   silicon  nusb, pollster            an external dependency
#   cloud    ureq                      an external dependency
#   gpu      wgpu, pollster            an external dependency
#   serve    two [[bin]] targets       an application, not a library capability
#   meter    std::process              an API the core's wasm target does not have
#
# A sibling with none of those is a sibling with no reason to be one, and being out there costs it
# eight surfaces. `ommx` had zero external dependencies, zero binaries and no wasm-hostile API; it
# is now src/ommx.rs beside ftp.rs and lp.rs.
echo
echo "── sibling crates: what keeps each one out of the core ─────────────────────"
orphans=0
# `set -euo pipefail` is on, and a grep that matches nothing returns 1 -- which under pipefail kills
# the whole substitution and, with -e, the script. The first version of this block printed its
# heading and silently stopped. Counting through a function that always succeeds is the fix.
# `grep -c` PRINTS a count and RETURNS 1 when that count is zero, so `|| echo 0` appended a second
# zero and the integer test below silently failed on "0\n0". Take grep's own number and only
# default when there is none.
count_matches() { local n; n=$(grep -cE "$1" "$2" 2>/dev/null || true); echo "${n:-0}"; }
for c in $(ls -d */ 2>/dev/null | tr -d /); do
  [[ -f "$c/Cargo.toml" ]] || continue
  grep -q '^\[package\]' "$c/Cargo.toml" || continue
  deps_block=$(sed -n '/^\[dependencies\]/,/^\[/p' "$c/Cargo.toml" 2>/dev/null || true)
  ext=$(printf '%s\n' "$deps_block" | grep -E '^[a-z0-9_-]+ *=' 2>/dev/null | { grep -vc '^ferrotherm' 2>/dev/null || true; })
  ext=${ext:-0}
  bins=$(count_matches '^\[\[bin\]\]' "$c/Cargo.toml")
  # Third instance of the same trap in one block, so it is worth naming: under `set -euo pipefail`
  # a grep that matches nothing returns 1, pipefail propagates that through the pipeline, and -e
  # kills the script AFTER the assignment has already succeeded. Every one of these needs its own
  # `|| true`, and the symptom is a heading printed with nothing under it.
  hostile=$( { grep -rl 'std::process\|std::net\|std::thread' "$c/src" 2>/dev/null || true; } | wc -l | tr -d ' ')
  if [[ "${ext:-0}" -gt 0 ]]; then
    printf '  %-10s %s external dependenc(ies)\n' "$c" "$ext"
  elif [[ "${bins:-0}" -gt 0 ]]; then
    printf '  %-10s an application (%s binary target(s))\n' "$c" "$bins"
  elif [[ "${hostile:-0}" -gt 0 ]]; then
    printf '  %-10s uses std APIs the core wasm target does not have\n' "$c"
  else
    printf '  %-10s NOTHING KEEPS IT OUT -- it belongs in the core\n' "$c"
    orphans=$((orphans + 1))
  fi
done

echo
echo "checked ${#symbols[@]} C ABI symbols across ${#SURFACES[@]} surfaces ($exempted documented gaps)"
if [[ $stale -gt 0 ]]; then
  echo
  echo "$stale EXEMPT entries name symbols that do not exist. Remove them." >&2
  exit 1
fi
if [[ $orphans -gt 0 ]]; then
  echo
  echo "$orphans sibling crate(s) have no reason to be siblings." >&2
  echo "A crate outside the core cannot be reached by the C ABI, so it reaches Rust and none of the" >&2
  echo "other eight surfaces. Move it into the core, or give it the dependency that justifies it." >&2
  exit 1
fi
if [[ $missing -gt 0 ]]; then
  echo
  echo "$missing symbol/surface pairs are unreachable." >&2
  echo "Either bind them, or add a line to EXEMPT in this script WITH A REASON." >&2
  exit 1
fi
echo "every symbol reaches every surface that should have it"
