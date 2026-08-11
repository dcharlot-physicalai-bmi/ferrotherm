#!/usr/bin/env bash
#
# One version, five files.
#
# The library, the server, the Python package, the Julia package and the JLL all carry a version
# string, and they are released together out of this repository. Nothing checked they agreed, which
# is the shape of drift that has already cost this project twice: a value meaning two things in two
# places, and a wasm on the site that was not the wasm in the repo. A binding whose version says
# nothing about the library underneath it is a version nobody can use.
#
#   scripts/check-versions.sh

set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

# name:file:pattern — each extracts the version from its own format
read_version() {
  case "$1" in
    Cargo.toml|serve/Cargo.toml|julia/Ferrotherm/Project.toml|julia/ferrotherm_jll/Project.toml)
      grep -m1 '^version' "$1" | cut -d'"' -f2 ;;
    python/pyproject.toml)
      grep -m1 '^version' "$1" | cut -d'"' -f2 ;;
    python/ferrotherm/__init__.py)
      grep -m1 '^__version__' "$1" | cut -d'"' -f2 ;;
  esac
}

files=(
  Cargo.toml
  python/pyproject.toml
  python/ferrotherm/__init__.py
  julia/Ferrotherm/Project.toml
  julia/ferrotherm_jll/Project.toml
)

want="$(read_version Cargo.toml)"
if [[ -z "$want" ]]; then
  echo "could not read a version out of Cargo.toml" >&2
  exit 2
fi

bad=0
for f in "${files[@]}"; do
  got="$(read_version "$f")"
  if [[ "$got" == "$want" ]]; then
    printf '  %-36s %s\n' "$f" "$got"
  else
    printf '  %-36s %s   <- expected %s\n' "$f" "${got:-<none>}" "$want"
    bad=1
  fi
done

# The server versions independently -- it is a separate crate with its own release cadence -- but
# the ferrotherm it depends on must be the one in this repository, or `cargo publish` resolves to
# whatever is already on crates.io and quietly ships against an older library.
dep="$(grep -m1 '^ferrotherm = ' serve/Cargo.toml | sed -E 's/.*version = "([^"]+)".*/\1/')"
major_minor="${want%.*}"
if [[ "$dep" == "$major_minor" || "$dep" == "$want" ]]; then
  printf '  %-36s depends on ferrotherm %s\n' "serve/Cargo.toml" "$dep"
else
  printf '  %-36s depends on ferrotherm %s   <- expected %s\n' "serve/Cargo.toml" "$dep" "$major_minor"
  bad=1
fi

if [[ $bad -eq 1 ]]; then
  echo
  echo "these are released together; a version that disagrees is a version nobody can use." >&2
  exit 1
fi
echo "all agree on $want"
