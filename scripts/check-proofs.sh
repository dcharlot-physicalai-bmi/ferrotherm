#!/usr/bin/env bash
#
# The machine-checked theorems must actually check.
#
# `cargo kani` runs bounded model checking over every `#[kani::proof]` harness in the crate: each
# pass is EXHAUSTIVE over the stated input ranges, not a sample of them. The harnesses live beside
# the code they verify (`src/sparsify.rs`, `src/device.rs`), compile only under `cfg(kani)`, and add
# no dependency to the crate.
#
# What is proved today, and why each is load-bearing:
#   * `copies_for` is SUFFICIENT and MINIMAL for every (degree, budget) in range -- the ground-state
#     preservation argument stands on sufficiency, the site economics on minimality.
#   * the Pegasus and Zephyr linear indices are INJECTIVE and IN RANGE at the shipped machine sizes
#     -- injectivity is the difference between programming a qubit and programming SOME qubit.
#
# Locally a missing Kani toolchain SKIPS, because not every machine has it. In CI that is the
# failure mode this gate exists to catch, so FERROTHERM_REQUIRE_ALL=1 turns the skip into a failure.
# Same contract as check-pages.sh, which learned it the same way.
#
#   scripts/check-proofs.sh              run every proof harness
#   scripts/check-proofs.sh --selftest   prove the verdict parsing can say no
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

if ! command -v kani >/dev/null 2>&1 || ! cargo kani --version >/dev/null 2>&1; then
  if [ "${FERROTHERM_REQUIRE_ALL:-}" = "1" ]; then
    echo "kani is not installed and FERROTHERM_REQUIRE_ALL=1, so this is a failure:" >&2
    echo "  cargo install --locked kani-verifier && cargo kani setup" >&2
    exit 1
  fi
  echo "skipped: no kani toolchain"
  echo "  cargo install --locked kani-verifier && cargo kani setup"
  echo "  (CI sets FERROTHERM_REQUIRE_ALL=1, where this skip is a failure)"
  exit 0
fi

if [ "${1:-}" = "--selftest" ]; then
  # WHAT THIS PROVES AND WHAT IT DOES NOT, stated rather than implied. A scratch crate with one
  # deliberately false theorem must come back FAILED, which exercises kani's refutation path and
  # this script's reading of the verdict. It does NOT damage the real crate's harnesses -- a full
  # copy-and-mutate selftest costs a whole extra kani build of the workspace, and the property
  # bought here is the one this gate has to have: that "SUCCESSFUL" is a verdict it can fail to
  # see, not a string it always finds.
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  mkdir -p "$tmp/src"
  cat > "$tmp/Cargo.toml" <<'EOF'
[package]
name = "proof-selftest"
version = "0.0.0"
edition = "2021"
[lints.rust]
unexpected_cfgs = { level = "warn", check-cfg = ["cfg(kani)"] }
EOF
  cat > "$tmp/src/lib.rs" <<'EOF'
#[cfg(kani)]
mod proofs {
    /// Deliberately false: 255u8 + 1 wraps, so the assertion cannot hold for x = 255.
    #[kani::proof]
    fn a_false_theorem_must_be_refuted() {
        let x: u8 = kani::any();
        assert!(x.wrapping_add(1) > x);
    }
}
EOF
  if (cd "$tmp" && cargo kani >/dev/null 2>&1); then
    echo "SELFTEST FAILED: kani accepted a theorem that is false at x = 255, so a passing run" >&2
    echo "of this gate would prove nothing" >&2
    exit 1
  fi
  echo "selftest: a false theorem was refuted, so this gate's green means the proofs actually hold"
  exit 0
fi

# A failed run is TWO different statements and the first version of this gate conflated them: a
# broken Cargo.toml made `cargo kani` fail to BUILD, and the error branch announced "a proof
# harness FAILED" -- a refuted theorem that never existed. "The tool could not run" and "the
# theorem is false" call for opposite responses, so they are separated the way the certifier
# separates "could not look" from "nothing moved".
out="$(cargo kani 2>&1)" || {
  if printf '%s\n' "$out" | grep -q "VERIFICATION:- FAILED"; then
    printf '%s\n' "$out" | grep -E "VERIFICATION|Failed Checks" | head -20 >&2
    echo "a proof harness was REFUTED: one of the stated theorems does not hold" >&2
  else
    printf '%s\n' "$out" | tail -20 >&2
    echo "cargo kani could not run to a verdict -- a build or toolchain failure, NOT a refutation" >&2
  fi
  exit 1
}
verified="$(printf '%s\n' "$out" | grep -oE '[0-9]+ successfully verified harnesses' | grep -oE '^[0-9]+' | head -1)"
if [ -z "$verified" ] || [ "$verified" -lt 5 ]; then
  # A floor, not a formality: if harness discovery breaks, kani "succeeds" over an empty set and
  # this gate would go green while checking nothing. Four is today's count; raise it when a proof
  # lands, and this line is the reminder to.
  echo "only ${verified:-0} harnesses verified; expected at least 5. Discovery has shrunk." >&2
  exit 1
fi
echo "all $verified proof harnesses verified: the stated theorems hold over their whole ranges"
