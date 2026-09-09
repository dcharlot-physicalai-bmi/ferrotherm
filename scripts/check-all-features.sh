#!/usr/bin/env bash
#
# Every gate this repository owns, run again with EVERY FEATURE ON.
#
# WHY, with the incident. `ferrotherm-silicon` declares one optional feature, `flash` -- 481 lines
# of FT2232H MPSSE JTAG that flashes a real FPGA. No gate anywhere passed `--features flash` or
# `--all-features`, so for as long as it existed:
#
#   * `cargo build -p ferrotherm-silicon --features flash` failed with 23 missing-documentation
#     errors. A SHIPPED CRATE HAD A FEATURE THAT DID NOT COMPILE.
#   * `cargo clippy` with it on failed with 27, including thirteen fallible functions with no
#     `# Errors` section and a `# Panics`-free assert.
#   * three of the crate's own examples did not compile, for want of the `allow(missing_docs)`
#     header every example in the core crate carries.
#   * `cargo test` could not build the test binary at all, so its tests had never run.
#
# The crate's own doc-coverage gate reported 100%. It was measuring the default-feature surface,
# which is the surface that does not contain any of the above. A lint run without the feature flag
# misses everything behind it, and reports a clean number for the part it can see.
#
# WHAT THIS IS NOT. It is not a second test suite -- it is the SAME gates, on the whole surface.
# Anything that needs hardware still skips; `flash`'s tests are unit tests of the bit layouts and
# run anywhere.

set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

# Every feature in the workspace, listed so a reader can see what "all" is. One today.
feats=$(python3 - <<'PY'
import glob, re
out = []
for f in ["Cargo.toml"] + sorted(glob.glob("*/Cargo.toml")):
    s = open(f, encoding="utf-8").read()
    m = re.search(r"^\[features\](.*?)(^\[|\Z)", s, re.M | re.S)
    if m:
        for name in re.findall(r"^(\w[\w-]*)\s*=", m.group(1), re.M):
            if name != "default":
                out.append(f"{f.split('/')[0] if '/' in f else 'ferrotherm'}/{name}")
print(" ".join(out))
PY
)
echo "  features: ${feats:-none declared}"
if [ -z "$feats" ]; then
  echo "  no optional features in the workspace, so --all-features is the default surface"
  exit 0
fi

echo "  build"
cargo build --workspace --all-features --all-targets --quiet
echo "  clippy"
cargo clippy --workspace --all-features --all-targets --quiet -- -D warnings
echo "  rustdoc"
RUSTDOCFLAGS=-Dwarnings cargo doc --workspace --all-features --no-deps --quiet
echo "  tests"
cargo test --workspace --all-features --quiet
echo "  every gate passed with every feature on"
