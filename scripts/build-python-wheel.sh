#!/usr/bin/env bash
#
# Build a Python wheel with the compiled library inside it.
#
# `pip install ferrotherm` has to work with nothing else present: no cargo, no checkout, no
# FERROTHERM_LIB. So the wheel carries the cdylib for the platform it was built on, dropped into the
# package directory, which is the first place the loader looks after the environment variable.
#
# The wheel is therefore platform-specific and one build produces one platform. CI runs this across
# the matrix; run it locally to check the shape.
#
#   scripts/build-python-wheel.sh [--test]

set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

case "$(uname -s)" in
  Darwin) lib=libferrotherm.dylib ;;
  Linux)  lib=libferrotherm.so ;;
  *)      lib=ferrotherm.dll ;;
esac

echo "building the library"
cargo build --release
cp "target/release/$lib" "python/ferrotherm/$lib"

# The platform tag has to describe the LIBRARY, not the interpreter that happens to be running.
# A universal2 CPython tags the wheel universal2 even when the dylib inside is arm64 only, and that
# wheel installs on an Intel Mac and fails at import. Read the architecture out of the binary.
plat=""
if [[ "$(uname -s)" == "Darwin" ]]; then
  archs="$(lipo -archs "python/ferrotherm/$lib")"
  case "$archs" in
    *arm64*x86_64*|*x86_64*arm64*) plat="macosx_11_0_universal2" ;;
    arm64)                         plat="macosx_11_0_arm64" ;;
    x86_64)                        plat="macosx_10_12_x86_64" ;;
    *) echo "unrecognised architecture in $lib: $archs" >&2; exit 1 ;;
  esac
  echo "  $lib is $archs -> $plat"
fi

echo "building the wheel"
cd python
rm -rf build dist ./*.egg-info
if [[ -n "$plat" ]]; then
  python3 -m build --wheel -C--build-option=--plat-name="$plat"
else
  python3 -m build --wheel
fi

# The library is a build artefact and does not belong in the tree afterwards: leaving it there makes
# the checkout behave differently from a fresh clone, which is how a "works on my machine" starts.
rm -f "ferrotherm/$lib"
ls -la dist/

if [[ "${1:-}" == "--test" ]]; then
  echo
  echo "installing the wheel into a clean environment"
  tmp="$(mktemp -d)"
  python3 -m venv "$tmp/venv"
  "$tmp/venv/bin/pip" install -q --upgrade pip
  "$tmp/venv/bin/pip" install -q dist/*.whl pytest

  # The point of a wheel is that it needs nothing else. Copy the tests OUT of the checkout and run
  # them somewhere the source cannot be reached, with FERROTHERM_LIB unset -- otherwise the loader
  # falls back to target/release and the suite passes while proving nothing about the wheel.
  cp "$here/python/test_model.py" "$tmp/test_model.py"
  # strip the sys.path line that points at the source tree
  python3 - "$tmp/test_model.py" <<'PY'
import re, sys
p = sys.argv[1]
t = open(p).read()
t = t.replace('sys.path.insert(0, str(Path(__file__).resolve().parent))\n', '')
open(p, 'w').write(t)
PY

  echo "checking the wheel is self-contained"
  ( cd "$tmp" && env -u FERROTHERM_LIB "$tmp/venv/bin/python" - <<'PY'
import ferrotherm, pathlib, sys
lib = pathlib.Path(ferrotherm.library_path()).resolve()
pkg = pathlib.Path(ferrotherm.__file__).resolve().parent
print("  package  ", pkg)
print("  library  ", lib)
if lib.parent != pkg:
    sys.exit(f"the wheel loaded a library from OUTSIDE itself ({lib}); it is not self-contained")
print("  the library came from inside the wheel")
PY
  )

  echo "running the tests against the installed wheel"
  ( cd "$tmp" && env -u FERROTHERM_LIB "$tmp/venv/bin/python" -m pytest -q test_model.py )
  rm -rf "$tmp"
fi
