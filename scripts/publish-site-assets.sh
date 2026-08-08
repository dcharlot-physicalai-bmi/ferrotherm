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

# Every symbol the pages call across the boundary. A page that loads but silently no-ops because an
# export went missing is exactly the failure this catches.
required_exports=(
  ft_model_new ft_model_free ft_model_categorical ft_model_integer ft_model_binary
  ft_model_name ft_model_not_equal ft_model_equal ft_model_fix
  ft_model_cardinality ft_model_at_most ft_model_at_least
  ft_model_objective_term ft_model_objective_pair
  ft_model_compile ft_model_solve ft_model_solve_with
  ft_model_value ft_model_feasible ft_model_energy ft_model_penalty
  ft_model_error ft_model_ftp ft_scratch
  ft_model_violations ft_model_violation
)

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

missing=()
for sym in "${required_exports[@]}"; do
  grep -q "$sym" "$site/ferrotherm.wasm" 2>/dev/null || missing+=("$sym")
done
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "the published wasm is missing exports the pages call: ${missing[*]}" >&2
  exit 1
fi
echo "  all ${#required_exports[@]} exports the pages call are present"

if [[ $stale -eq 1 ]]; then
  echo
  echo "the site is behind this repository. Run scripts/publish-site-assets.sh to refresh," >&2
  echo "then deploy the site." >&2
  exit 1
fi
