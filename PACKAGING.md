# Packaging

Where each surface is published, and what it takes.

| surface | registry | state |
|---|---|---|
| Rust library | crates.io | **`ferrotherm` 0.7.0** — published |
| Agent server | crates.io | **`ferrotherm-serve` 0.3.0** — published |
| Python | PyPI | wheels build and pass; needs a Trusted Publisher configured once |
| Julia | General | blocked on a `ferrotherm_jll`, see below |
| Zig | package index | not started |
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

`.github/workflows/python-release.yml` builds macOS arm64 and x86_64, Linux x86_64 (relabelled to
`manylinux_2_17` by auditwheel, which has nothing to vendor because the crate has no dependencies),
and Windows x86_64. Every wheel runs the self-containment check on its own runner before upload.

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

Zig's package manager wants a `build.zig` and a `build.zig.zon` with a content hash of a release
tarball. Not started. It has the same native-library question and the cleanest answer there is a
`build.zig` that invokes cargo, since a Zig consumer already has a build step.
