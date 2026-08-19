# Releasing

Six crates ship out of this repository and they are **not** independent: five of them pin the core
library, and the Python, Julia and JLL packages carry the core's version string in their own
manifests. `scripts/check-versions.sh` enforces both halves of that, and CI runs it.

This file exists because it did not. `ferrotherm-gpu` 0.2.0 sat in the tree, changelogged and
tagged, while crates.io served 0.1.0 — nobody had written the order down, so `cargo publish` was
something you remembered rather than something you followed. A release procedure that lives only in
someone's head is a release procedure that skips a crate.

## The order

Two orderings matter, and one of them is not obvious.

**Publish before you push.** `cargo publish` needs the version bump *committed*, not *pushed* — and
`check-versions.sh` fails when a pushed commit is ahead of crates.io, which is exactly right and
exactly what it did to the 0.13.0 release: CI started the moment the commit landed, ran during the
ninety seconds the five dependents were still uploading, and correctly reported four crates ahead.
Nothing was wrong except the order. Commit, publish all six, *then* push, and the window does not
exist.

**Core first.** crates.io resolves dependencies at publish time, so the core must be **live** before
anything that pins it can go out. The index takes a few seconds to catch up.

```sh
scripts/check-versions.sh          # everything agrees, and what is not yet on crates.io
cargo test --release --workspace
cargo clippy --release --workspace --all-targets -- -D warnings
scripts/check-parity.sh            # every C ABI symbol reaches every binding
scripts/check-semantics.sh         # every binding compiles one model to the same bytes
scripts/check-answers.sh           # ...and solves it to the same answer, which is not the same check
scripts/check-exports.sh           # every exported binding name resolves

cargo publish -p ferrotherm        # FIRST. Everything below pins it.
cargo publish -p ferrotherm-gpu
cargo publish -p ferrotherm-meter
cargo publish -p ferrotherm-cloud
cargo publish -p ferrotherm-silicon
cargo publish -p ferrotherm-serve

scripts/check-versions.sh          # and now it should say all six are live

git push origin main               # LAST, so CI never sees a half-published release
git push origin vX.Y.Z
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

## The site carries its own copy of the browser surfaces

`v2/` is a **separate repository**, so no ferrotherm CI job can see whether the deployed editor
matches the code. That is not an oversight to fix with more CI — it is a fact about where the files
live, and the consequence is that this is a step someone has to run:

```sh
scripts/publish-site-assets.sh --check   # exits 1 if the site is behind
scripts/publish-site-assets.sh           # rebuild, copy, verify
```

Then commit the result **in `v2/`**, which is a different repository with a different remote.

Skipping it is not visibly broken, which is exactly the problem. The deployed wasm was found several
releases behind and missing `ft_model_ommx`: the editor loaded, the button was there, and clicking it
did nothing, because a missing export is `undefined` in JavaScript rather than an error. A stale
surface that *works* is worse than one that breaks.

## After crates.io

- **Python**: `.github/workflows/python-release.yml` builds the wheels; it fires on a `v*` tag.
- **Julia**: `scripts/build-julia-artifacts.jl`, and the JLL manifest must point at the **published**
  release URLs. It once pointed at an expiring CI artifact, so `ferrotherm_jll` was never installable
  by anyone; the release job now attaches the tarballs and verifies the URLs it wrote.
