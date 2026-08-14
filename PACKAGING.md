# Packaging

Where each surface is published, and what it takes.

| surface | registry | state |
|---|---|---|
| Rust library | crates.io | **`ferrotherm` 0.8.0** |
| Agent server | crates.io | **`ferrotherm-serve` 0.4.0** |
| Python | PyPI | **`ferrotherm` 0.8.0** — three wheels |
| Julia | our own artifacts | `ferrotherm_jll` built and loading; registry is the open question |
| Zig | package index | `zig build test` builds the Rust library and runs 17 tests; not submitted |
| Browser | Institute site | live, `scripts/publish-site-assets.sh` |

## The thing all of them share

This is a native library with bindings, so every binding has the same question: **where does the
compiled library come from?** A binding that installs and then cannot find its library is worse
than one that does not install, because it fails later, further from the cause, and looks like the
user's fault.

## Python

`scripts/build-python-wheel.sh --test` builds a wheel with the library inside it and then proves
that claim: it installs the wheel into a clean virtualenv, copies the tests out of the checkout,
unsets `FERROTHERM_LIB`, and asserts the loaded library sits **inside site-packages** before
running anything. Without that assertion the suite passes on a wheel that quietly loaded
`target/release`, which is no test at all.

Two things the default wheel tagging gets wrong, both fixed in `python/setup.py`:

- The package is ctypes, not a C extension, so it runs on any Python 3. Making the wheel
  platform-specific also made setuptools tag it `cp39-cp39`, one wheel per Python version for no
  reason. It is `py3-none-<platform>`.
- The platform tag must describe the **library**, not the interpreter. A universal2 CPython tagged
  an arm64-only dylib as `universal2` — a wheel that installs on an Intel Mac and fails at import.
  The build script reads the architecture out of the binary with `lipo` and passes `--plat-name`.

`.github/workflows/python-release.yml` builds macOS arm64, Linux x86_64, and Windows x86_64.

Its filename is load-bearing. PyPI's Trusted Publisher is bound to `python-release.yml` in this
repository, so renaming the file makes PyPI reject the OIDC token with an error about a workflow it
has never heard of. It builds the Julia artifacts as well, despite the name. Every wheel runs the self-containment check on the platform it targets before upload.

**Verified.** A build-only run produced all three wheels, each carrying its library and tagged for
exactly where it runs:

```
ferrotherm-0.7.0-py3-none-macosx_11_0_arm64.whl        ferrotherm/libferrotherm.dylib
ferrotherm-0.7.0-py3-none-manylinux_2_28_x86_64.whl    ferrotherm/libferrotherm.so
ferrotherm-0.7.0-py3-none-win_amd64.whl                ferrotherm/ferrotherm.dll
```

macOS is Apple Silicon only. Intel Macs are outside Apple's own support window, and their runners
are scarce enough to stall a release for the best part of an hour. Anyone still on one can build the
library and set `FERROTHERM_LIB`, which is the documented fallback and works.

Linux gets its own job, inside a `manylinux_2_28` container. A `.so` built on the ubuntu-latest
runner links that runner's glibc, and auditwheel can only relabel a wheel **down to a policy the
binary already satisfies** — it refused ours outright:

> cannot repair to `manylinux_2_17_x86_64` because of the presence of too-recent versioned
> symbols. You'll need to compile the wheel on an older toolchain.

So we compile on an older toolchain. glibc 2.28 is RHEL 8, which every current distribution
satisfies.

**To publish, once:** configure a PyPI Trusted Publisher. No API token goes in this repository —
a token that leaks can publish anything, where a trusted publisher is bound to one workflow in one
repository. See the checklist at the end.

## Julia

Julia has no wheels. A package needing a native library depends on a **JLL**: a package carrying
prebuilt binaries per platform, which hands you a path.

The usual route to one is a recipe submitted to
[Yggdrasil](https://github.com/JuliaPackaging/Yggdrasil), built and reviewed by other people. We
build our own instead, out of the same CI job that builds the Python wheels, because the whole
point of this stack is that we own every layer of it.

Nothing about that requires BinaryBuilder. A `git-tree-sha1` and a `sha256` are computable with
stock Julia, and Rust already cross-compiles, so `scripts/build-julia-artifacts.jl` takes the
libraries the wheel jobs already produced, tars each one, hashes it, and writes the
`Artifacts.toml` that names them.

**Self-hosting costs nothing in trust.** `Artifacts.toml` names each tarball by URL *and by hash*,
and Julia refuses anything that does not match. The hash lives in the package; only the bytes live
on the release. That is the same guarantee a registry-hosted artifact gives.

The release job proves the artifact is the library it claims before publishing anything: it serves
the tarballs locally, loads the JLL against them, and calls `ft_onsager(0.5)`, requiring
`0.911319377877496` — Onsager's closed form, to the last digit.

`Ferrotherm.jl` uses the JLL when it is installed and falls back to a checkout otherwise, in this
order: `FERROTHERM_LIB` (an explicit override always wins — someone who set it is debugging a
specific build), then the artifact, then `target/release`. It is a **soft** dependency on purpose:
a hard one would mean `Ferrotherm` could not be installed without a registry carrying both, and the
package is useful today from a repository URL.

### The open question

One-step install (`Pkg.add("Ferrotherm")`) needs a registry, and a Julia registry must be the root
of its own repository — it cannot live in a subdirectory of this one. So the choice is:

- **Our own registry repo.** `Pkg.Registry.add("https://github.com/.../IPAIRegistry")` once, then
  `Pkg.add("Ferrotherm")` forever. Entirely ours, no third party, no review queue.
- **Leave it.** Users install from the URL, which works today and needs nothing.

Not General: registering there means submitting to their AutoMerge, which is the same posture we
declined for Yggdrasil.

## Zig

`zig/build.zig` answers the native-library question by building it: `cargo build --release` is a
build step, and the module links against what it produces with an rpath so the test binary starts
without `LD_LIBRARY_PATH`. No prebuilt artefact to go stale, no environment variable, and the
library always matches the source beside it.

```
zig build test          # builds the Rust library, then runs the binding's 17 tests
zig build -Dcargo=false # link only, if you build the Rust side yourself
```

Not submitted anywhere. Zig's package index takes a URL and a content hash of a release tarball,
which is a step to take once the tags settle.
