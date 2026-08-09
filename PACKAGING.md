# Packaging

Where each surface is published, and what it takes.

| surface | registry | state |
|---|---|---|
| Rust library | crates.io | **`ferrotherm` 0.7.0** — published |
| Agent server | crates.io | **`ferrotherm-serve` 0.3.0** — published |
| Python | PyPI | wheels build and pass; needs a Trusted Publisher configured once |
| Julia | General | blocked on a `ferrotherm_jll`, see below |
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

`.github/workflows/python-release.yml` builds macOS arm64 and x86_64, Linux x86_64, and Windows
x86_64. Every wheel runs the self-containment check on the platform it targets before upload.

**Verified.** A build-only run produced three of the four wheels, each carrying its library and
tagged for exactly where it runs:

```
ferrotherm-0.7.0-py3-none-macosx_11_0_arm64.whl        ferrotherm/libferrotherm.dylib
ferrotherm-0.7.0-py3-none-manylinux_2_28_x86_64.whl    ferrotherm/libferrotherm.so
ferrotherm-0.7.0-py3-none-win_amd64.whl                ferrotherm/ferrotherm.dll
```

The fourth, `macos-x86_64`, builds on a `macos-13` runner and those are scarce — it queued for
over twenty minutes without starting. It is not stuck and it is not broken; Intel Mac runners are
simply in short supply, and a release triggered by a tag will wait for one. The alternative is
cross-compiling from the arm64 runner, which would build the wheel in seconds and then be unable to
RUN it, since GitHub's Apple Silicon runners have no Rosetta. A wheel tested on the hardware it
targets is worth waiting for.

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

Blocked, and the blocker is real rather than procedural.

General's AutoMerge **tests that the package can be imported** on a clean machine. `Ferrotherm.jl`'s
`__init__` searches for the native library and calls `error(...)` when it finds none, so
`using Ferrotherm` fails there and registration is refused.

The standard answer is a **JLL package**: `ferrotherm_jll`, built by
[Yggdrasil](https://github.com/JuliaPackaging/Yggdrasil), which cross-compiles the cdylib for every
platform Julia supports and ships it as an artifact. `Ferrotherm.jl` then depends on it and
`using Ferrotherm` works everywhere with nothing installed by hand.

The alternative — softening `__init__` to a warning so the import succeeds — would register a
package that installs cleanly and then fails at the first call. That is precisely the failure this
project spent a week removing: a surface that looks fine and does nothing. Not doing that.

Building the JLL means a pull request to Yggdrasil, a third-party repository. That is a decision
for the Institute to make and not one to take unilaterally, so it is written down here rather than
opened.

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
