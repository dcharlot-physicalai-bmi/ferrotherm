#!/bin/sh
# The harness dlopens NVML. On aarch64, c_char is unsigned, and a signature
# written as *const i8 fails to typecheck there while passing on x86_64.
# `cargo check` does not link, so this does not need an aarch64 sysroot.
set -eu
cd "$(dirname "$0")/.."
# The pin in rust-toolchain.toml, not a hardcoded older compiler. c_char's
# signedness is a property of the target, so the check has to be the toolchain
# the workspace actually builds with.
channel=$(sed -n 's/^channel = "\([^"]*\)".*/\1/p' rust-toolchain.toml | head -1)
if [ -n "$channel" ] && cargo "+$channel" --version >/dev/null 2>&1; then
  rustup target add aarch64-unknown-linux-gnu --toolchain "$channel"
  exec cargo "+$channel" check -p ferrotherm-thrml-bench --target aarch64-unknown-linux-gnu
fi
rustup target add aarch64-unknown-linux-gnu
exec cargo check -p ferrotherm-thrml-bench --target aarch64-unknown-linux-gnu
