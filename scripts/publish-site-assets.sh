#!/usr/bin/env bash
#
# Publish the browser surfaces to the Institute site.
#
# The site serves these as static assets copied out of this repository. That copy was manual, which
# meant the live editor could sit several releases behind the code with nothing to say so -- and it
# did: the deployed wasm was missing every export added in a day's work while the page still loaded
# and still answered questions, just with the old vocabulary. A stale surface that WORKS is worse
# than one that breaks, because nobody goes looking.
#
# So this rebuilds, copies, and then verifies -- both that the bytes match and that the wasm really
# contains the symbols the page will call.
#
#   scripts/publish-site-assets.sh [--check]
#
# --check verifies without copying, and exits non-zero if the site is stale. That is the form to run
# in CI or before a deploy.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
site="${FERROTHERM_SITE_ASSETS:-$here/../../v2/public/assets/ferrotherm}"
check_only=0
[[ "${1:-}" == "--check" ]] && check_only=1

if [[ ! -d "$site" ]]; then
  echo "no site asset directory at $site" >&2
  echo "set FERROTHERM_SITE_ASSETS to point at it" >&2
  exit 2
fi

if [[ $check_only -eq 0 ]]; then
  echo "building wasm"
  (cd "$here" && cargo build --release --lib --target wasm32-unknown-unknown)
  cp "$here/target/wasm32-unknown-unknown/release/ferrotherm.wasm" "$here/docs/ferrotherm.wasm"
fi

stale=0
for f in graph.html ide.html index.html ferrotherm.wasm; do
  src="$here/docs/$f"
  dst="$site/$f"
  if [[ ! -f "$src" ]]; then
    echo "missing source $src" >&2
    exit 2
  fi
  if cmp -s "$src" "$dst" 2>/dev/null; then
    printf '  %-18s up to date\n' "$f"
  elif [[ $check_only -eq 1 ]]; then
    printf '  %-18s STALE on the site\n' "$f"
    stale=1
  else
    cp "$src" "$dst"
    printf '  %-18s copied (%s bytes)\n' "$f" "$(wc -c < "$src" | tr -d ' ')"
  fi
done

# The same check CI runs, against the copy the SITE will serve rather than the one in the repo.
"$here/scripts/check-wasm-exports.sh" "$site/ferrotherm.wasm" | sed 's/^/  /'


if [[ $stale -eq 1 ]]; then
  echo
  echo "the site is behind this repository. Run scripts/publish-site-assets.sh to refresh," >&2
  echo "then deploy the site." >&2
  exit 1
fi
