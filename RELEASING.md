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

**Publish before you push MAIN — but push the TAG first.** `cargo publish` needs the version bump
*committed*, not *pushed*, and `check-versions.sh` fails when a pushed commit is ahead of crates.io,
which is exactly right and exactly what it did to the 0.13.0 release: CI started the moment the
commit landed, ran during the ninety seconds the dependents were still uploading, and correctly
reported four crates ahead. Nothing was wrong except the order.

That advice used to say "push last", full stop, and then `cut-release.sh` arrived and **requires the
tag to be on origin** before it will publish anything (the registry must never lead the public
history). The two rules were never reconciled, and at 0.40.0 they collided: `git push origin main`
and `git push origin v0.40.0` went out together, CI on main started at 10:39:32, and
`ferrotherm-silicon 0.2.25` was still uploading. The job failed on ONE crate reading `<- ahead`,
with nothing wrong but the order — the 0.13.0 incident, a second time, from a document that had
described only half of the sequence.

So the order is three steps, not two: **push the tag, publish, then push main.** The tag push
triggers the release workflow, which does not check the registry; main is what CI reads.

**Core first.** crates.io resolves dependencies at publish time, so the core must be **live** before
anything that pins it can go out. The index takes a few seconds to catch up.

**And the core is no longer the only such edge.** `ferrotherm-gpu` took a dev-dependency on
`ferrotherm-meter` when the energy comparison needed both a GPU and a meter, which makes meter a
second crate that has to be live first. The list above had gpu before meter and was right until
that edge existed. Read the dependency graph rather than the list:

```sh
cargo metadata --no-deps --format-version 1 \
  | python3 -c 'import json,sys
p={p["name"]:p for p in json.load(sys.stdin)["packages"]}
for n,pk in sorted(p.items()):
    d=sorted({x["name"] for x in pk["dependencies"] if x["name"] in p})
    print(f"{n:26} needs {d}")'
```

```sh
scripts/check-versions.sh          # everything agrees, and what is not yet on crates.io
cargo test --release --workspace
cargo clippy --release --workspace --all-targets -- -D warnings
scripts/check-parity.sh            # every C ABI symbol reaches every binding
scripts/check-semantics.sh         # every binding compiles one model to the same bytes
scripts/check-answers.sh           # ...and solves it to the same answer on all seven surfaces
scripts/check-exports.sh           # every exported binding name resolves

git tag -a vX.Y.Z -m "..."          # on the release commit, still local
git push origin vX.Y.Z             # the TAG first: cut-release.sh refuses an unpushed one

scripts/cut-release.sh X.Y.Z       # publishes all six from a worktree of the tag, in dep order
                                   # (rerun-safe: a crate already on the registry is skipped)

scripts/check-versions.sh          # and now it should say all six are live

git push origin main               # LAST, so CI never sees a half-published release
```

By hand, if `cut-release.sh` is unavailable, the publish order is the dependency order:

```sh
cargo publish -p ferrotherm        # FIRST. Everything below pins it.
cargo publish -p ferrotherm-meter  # before gpu -- see below
cargo publish -p ferrotherm-gpu
cargo publish -p ferrotherm-cloud
cargo publish -p ferrotherm-silicon
cargo publish -p ferrotherm-serve
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
