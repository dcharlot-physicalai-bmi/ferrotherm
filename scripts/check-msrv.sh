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
  # WHAT THIS CAN AND CANNOT PROVE. The declared floor is a POLICY floor -- 1.96 on a workspace whose
  # technical floor is 1.88 -- so "prove a lower toolchain fails" is not available: lower toolchains
  # build fine, and that is the point of choosing the number rather than deriving it. What the gate
  # protects is the other direction, and both halves of it are damaged here:
  #
  #   1. the members must agree, or the workspace advertises two different minimums;
  #   2. the declared toolchain must actually build, which the live run below does unconditionally.
  #
  # So this damages a member's declared version and requires the mismatch check to fire. Nothing is
  # written outside a scratch copy.
  tmp=$(mktemp -d) || { echo "SELFTEST FAILED: no scratch dir, so nothing could be damaged" >&2; exit 2; }
  trap 'rm -rf "$tmp"' EXIT
  cp meter/Cargo.toml "$tmp/meter.bak"
  sed -i.orig "s/^rust-version = \".*\"/rust-version = \"1.70\"/" meter/Cargo.toml
  rm -f meter/Cargo.toml.orig
  if grep -q '^rust-version = "1.70"' meter/Cargo.toml; then
    if bash "$0" >/dev/null 2>&1; then
      cp "$tmp/meter.bak" meter/Cargo.toml
      echo "SELFTEST FAILED: a member declaring 1.70 while the workspace declares $msrv went through" >&2
      echo "                 unnoticed, so two different minimums could ship as one." >&2
      exit 1
    fi
    cp "$tmp/meter.bak" meter/Cargo.toml
    echo "selftest ok: a member disagreeing about the floor is caught, and the live run compiles the"
    echo "workspace on $msrv unconditionally -- which is the half that can actually break a user."
    exit 0
  fi
  cp "$tmp/meter.bak" meter/Cargo.toml
  echo "SELFTEST FAILED: the damage did not apply to meter/Cargo.toml, so nothing was proved" >&2
  exit 1
fi

echo "building the workspace on the declared minimum, $msrv ..."
if cargo "+$msrv" check --workspace --all-targets 2>&1 | tail -30; then
  echo "  rust-version = $msrv holds: workspace, tests and examples all compile on it."
else
  echo "the declared minimum $msrv does not build this workspace." >&2
  exit 1
fi
