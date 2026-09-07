#!/usr/bin/env bash
#
# Everything CI runs, in one command, before you push.
#
# There was no such command. Each gate is its own script, CI wires them together, and a developer
# ran whichever subset they happened to remember — which is a subset that drifts. It cost a red
# main: a Zig assertion contradicting its Rust counterpart shipped because the local loop being used
# that day ran `cargo test` and the shell gates and not `zig build test`, so CI was covering a blind
# spot in the development loop rather than duplicating it. Every green local run made that gap
# harder to see.
#
# So this is derived from `.github/workflows/ci.yml` and is meant to be a SUPERSET of it. When a
# step is added there, add it here.
#
#   scripts/preflight.sh            run every gate, keep going, report at the end
#   scripts/preflight.sh --fast     skip the slow ones (kani, mutation, release builds)
#
# It does NOT stop at the first failure. A run that halts on gate 3 of 20 tells you about gate 3 and
# hides the rest, and the whole point is to learn everything that is broken in one pass.
#
# A gate whose toolchain is missing SKIPS and says so, and skips are listed separately from passes
# at the end -- "I could not look" and "nothing is wrong" are different answers, and only one of
# them means you can push.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

fast=0
[ "${1:-}" = "--fast" ] && fast=1

log_dir="$(mktemp -d)"
trap 'rm -rf "$log_dir"' EXIT

passed=() ; failed=() ; skipped=()

# Run one gate. `need` is a command that must exist, or the gate is skipped by name.
gate() {
  local name="$1" need="${2:-}" ; shift 2
  if [ -n "$need" ] && ! command -v "$need" >/dev/null 2>&1; then
    skipped+=("$name (no $need)")
    printf '  \033[2m----\033[0m %-38s no %s\n' "$name" "$need"
    return
  fi
  local log="$log_dir/$name.log"
  if "$@" >"$log" 2>&1; then
    passed+=("$name")
    printf '  \033[32mok\033[0m   %s\n' "$name"
  else
    failed+=("$name")
    printf '  \033[31mFAIL\033[0m %s\n' "$name"
    # The tail immediately, because the summary at the end is a list of names and the thing you
    # actually want is the error.
    sed 's/^/         /' "$log" | tail -12
  fi
}

echo "preflight: everything CI runs${fast:+ (fast: slow gates skipped)}"
echo

echo "-- the crate ------------------------------------------------------------"
gate "tests"            cargo cargo test --workspace --quiet
gate "clippy"           cargo cargo clippy --workspace --all-targets -- -D warnings
gate "rustdoc"          cargo env RUSTDOCFLAGS=-Dwarnings cargo doc --workspace --no-deps
gate "examples-build"   cargo cargo build --workspace --examples --quiet
gate "msrv"             cargo scripts/check-msrv.sh
gate "versions"         cargo scripts/check-versions.sh

echo
echo "-- the bindings ---------------------------------------------------------"
gate "parity"           cargo scripts/check-parity.sh
gate "semantics"        cargo scripts/check-semantics.sh
gate "answers"          cargo scripts/check-answers.sh
gate "hubo-answers"     cargo scripts/check-hubo-answers.sh
gate "stubs"            python3 scripts/check-stubs.sh
gate "exports"          julia scripts/check-exports.sh
gate "zig"              zig sh -c 'cd zig && zig build test --summary all'
gate "wasm-exports"     cargo scripts/check-wasm-exports.sh

echo
echo "-- the surfaces ---------------------------------------------------------"
gate "editor-parity"    cargo scripts/check-editor-parity.sh
gate "editor-model"     cargo scripts/check-editor-model.sh
gate "browser-cert"     node scripts/check-browser-certificate.sh
gate "pages"            node scripts/check-pages.sh

echo
echo "-- the expensive ones ---------------------------------------------------"
if [ "$fast" = "1" ]; then
  skipped+=("gpu (--fast)" "proofs (--fast)" "mutation (--fast)")
  echo "  ---- skipped by --fast: gpu, proofs, mutation"
else
  gate "gpu"            cargo cargo test -p ferrotherm-gpu --quiet
  gate "proofs"         kani scripts/check-proofs.sh
  # The mutation suite breaks the code on purpose and restores from git, so it refuses on a dirty
  # tree -- correctly. Reporting that as a FAILURE would be this script lying: "you have
  # uncommitted changes" and "your tests have no teeth" are different statements, and only one of
  # them should stop a push. So it is a SKIP with its reason, which the summary lists separately.
  if git diff --quiet && git diff --cached --quiet; then
    gate "mutation"     cargo scripts/mutation-suite.sh
  else
    skipped+=("mutation (tree is dirty; it restores from git)")
    printf '  \033[2m----\033[0m %-38s tree is dirty\n' "mutation"
  fi
fi

echo
echo "========================================================================="
printf 'passed %d   failed %d   skipped %d\n' "${#passed[@]}" "${#failed[@]}" "${#skipped[@]}"
if [ "${#skipped[@]}" -gt 0 ]; then
  echo
  echo "SKIPPED -- these were not checked, which is not the same as checked and fine:"
  printf '  %s\n' "${skipped[@]}"
fi
if [ "${#failed[@]}" -gt 0 ]; then
  echo
  echo "FAILED:"
  printf '  %s\n' "${failed[@]}"
  exit 1
fi
echo
echo "every gate that could run, ran, and passed"
