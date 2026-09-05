#!/usr/bin/env bash
#
# Does the crate actually build on the toolchain it CLAIMS to support?
#
# `rust-version` in Cargo.toml is a promise to every downstream user, and nothing in a normal build
# tests it: CI runs stable, the developer runs stable, and the declared floor drifts upward silently
# the first time anyone writes a newer construct. That is how a crate ends up advertising a minimum
# it has not compiled on since the day the line was typed.
#
# This compiles the WHOLE WORKSPACE, tests and examples included, on exactly the declared toolchain.
#
#   scripts/check-msrv.sh              build on the declared rust-version
#   scripts/check-msrv.sh --selftest   prove the check can fail
#
# Installing a pinned toolchain is the cost. Without it this says so and exits 2 rather than
# reporting success, because "I could not check" and "it builds" are different answers.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

msrv=$(grep -m1 '^rust-version' Cargo.toml | cut -d'"' -f2)
[[ -n "$msrv" ]] || { echo "no rust-version in Cargo.toml -- there is no claim to check" >&2; exit 1; }

# Every member has to agree, or the workspace advertises two different floors.
mismatch=0
for f in Cargo.toml silicon/Cargo.toml serve/Cargo.toml cloud/Cargo.toml gpu/Cargo.toml meter/Cargo.toml; do
  v=$(grep -m1 '^rust-version' "$f" | cut -d'"' -f2)
  if [[ "$v" != "$msrv" ]]; then
    echo "  $f declares rust-version $v, the workspace declares $msrv" >&2
    mismatch=1
  fi
done
[[ $mismatch -eq 0 ]] || { echo "the members disagree about the minimum toolchain" >&2; exit 1; }

command -v rustup >/dev/null 2>&1 || { echo "needs rustup to pin a toolchain; this run checked NOTHING" >&2; exit 2; }
if ! rustup toolchain list | grep -q "^${msrv}"; then
  echo "toolchain $msrv is not installed, so this run checked NOTHING." >&2
  echo "  rustup toolchain install $msrv --profile minimal" >&2
  exit 2
fi

if [[ "${1:-}" == "--selftest" ]]; then
  # Damage the claim, not the code: assert an MSRV the sources provably do not meet. 1.85 is the
  # edition-2024 floor and is REJECTED here by the let-chains in `embed` and `fabric`, so if this
  # passes, the check is not compiling what it says it is.
  if ! rustup toolchain list | grep -q '^1\.85'; then
    echo "SELFTEST FAILED: 1.85 is not installed, so nothing could be shown to fail" >&2
    exit 2
  fi
  if cargo +1.85 check --workspace --all-targets >/dev/null 2>&1; then
    echo "SELFTEST FAILED: the workspace built on 1.85, which the declared floor of $msrv says" >&2
    echo "                 it should not. Either the floor is too high or this check is looking" >&2
    echo "                 at the wrong thing." >&2
    exit 1
  fi
  echo "selftest ok: 1.85 is rejected by the compiler, so a too-low claim would be caught here."
  exit 0
fi

echo "building the workspace on the declared minimum, $msrv ..."
if cargo "+$msrv" check --workspace --all-targets 2>&1 | tail -30; then
  echo "  rust-version = $msrv holds: workspace, tests and examples all compile on it."
else
  echo "the declared minimum $msrv does not build this workspace." >&2
  exit 1
fi
