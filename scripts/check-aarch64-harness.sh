#!/bin/sh
# The harness dlopens NVML. On aarch64, c_char is unsigned, and a signature
# written as *const i8 fails to typecheck there while passing on x86_64.
# `cargo check` does not link, so this does not need an aarch64 sysroot.
set -eu
cd "$(dirname "$0")/.."
if cargo +1.96.0 --version >/dev/null 2>&1; then
  rustup target add aarch64-unknown-linux-gnu --toolchain 1.96.0
  exec cargo +1.96.0 check -p ferrotherm-thrml-bench --target aarch64-unknown-linux-gnu
fi
rustup target add aarch64-unknown-linux-gnu
exec cargo check -p ferrotherm-thrml-bench --target aarch64-unknown-linux-gnu
