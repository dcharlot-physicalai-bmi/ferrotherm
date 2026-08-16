# Releasing

Six crates ship out of this repository and they are **not** independent: five of them pin the core
library, and the Python, Julia and JLL packages carry the core's version string in their own
manifests. `scripts/check-versions.sh` enforces both halves of that, and CI runs it.

This file exists because it did not. `ferrotherm-gpu` 0.2.0 sat in the tree, changelogged and
tagged, while crates.io served 0.1.0 — nobody had written the order down, so `cargo publish` was
something you remembered rather than something you followed. A release procedure that lives only in
someone's head is a release procedure that skips a crate.

## The order

crates.io resolves dependencies at publish time, so the core must be **live** before anything that
pins it can go out. The index takes a few seconds to catch up.

```sh
scripts/check-versions.sh          # everything agrees, and what is not yet on crates.io
cargo test --release --workspace
scripts/check-parity.sh            # every C ABI symbol reaches every binding
scripts/check-semantics.sh         # every binding compiles one model to the same bytes
scripts/check-exports.sh           # every exported binding name resolves

cargo publish -p ferrotherm        # FIRST. Everything below pins it.
cargo publish -p ferrotherm-gpu
cargo publish -p ferrotherm-meter
cargo publish -p ferrotherm-cloud
cargo publish -p ferrotherm-silicon
cargo publish -p ferrotherm-serve

scripts/check-versions.sh          # and now it should say all six are live
```

## What has to move together

A core bump touches **five** version strings and **five** dependency pins. `check-versions.sh` finds
the dependents rather than listing them, because a list goes stale the moment someone adds a crate.

| file | what it carries |
|---|---|
| `Cargo.toml` | the core version — the one everything else follows |
| `python/pyproject.toml`, `python/ferrotherm/__init__.py` | the wheel, which wraps the same C ABI |
| `julia/Ferrotherm/Project.toml`, `julia/ferrotherm_jll/Project.toml` | the Julia package and its binary artifact |
| `{cloud,gpu,meter,serve,silicon}/Cargo.toml` | `ferrotherm = { version = "0.N" }` |

A sibling crate that is **already on crates.io** needs its own version bumped too when its
dependency pin changes, or there is no new version to publish.

## When a crate should not be published

Set `publish = false` in its `Cargo.toml`. That is a statement, and the gate reads it. A crate
carrying a description, licence, keywords and categories but no publish — which is what
`ferrotherm-cloud` was for months — is the one state that says neither.

## After crates.io

- **Python**: `.github/workflows/python-release.yml` builds the wheels; it fires on a `v*` tag.
- **Julia**: `scripts/build-julia-artifacts.jl`, and the JLL manifest must point at the **published**
  release URLs. It once pointed at an expiring CI artifact, so `ferrotherm_jll` was never installable
  by anyone; the release job now attaches the tarballs and verifies the URLs it wrote.
