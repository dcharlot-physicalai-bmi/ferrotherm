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
# THE MANIFESTS THIS REPOSITORY OWNS, and nothing else.
#
# `find .` and `grep -r .` were both wrong here, and wrong in a way that only shows up on someone
# else's machine: they descend into NESTED CHECKOUTS. A git worktree lives under .claude/worktrees
# when an agent is given one, and a worktree is a full copy of the tree at whatever commit it was
# cut from -- so eight worktrees pinned one commit back reported all six crates as "registry is
# AHEAD of this tree" seconds after a correct release. Vendored copies and nested clones do the same.
#
# `git ls-files` enumerates what this repository TRACKS, which is exactly the question: a worktree
# is not tracked by its parent, and neither is target/. The fallback keeps the gate working outside
# a git checkout -- a published tarball, say -- where nested checkouts cannot exist anyway.
owned_manifests() {
  if git -C "$here" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "$here" ls-files -- '*Cargo.toml' | sed 's|^|./|'
  else
    find . -name Cargo.toml -not -path './target/*' -not -path './.claude/*'
  fi
}

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

# Placed AFTER `bad=0`, and that is not incidental: the first version of this gate sat above it,
# so the assignment reset the flag it had just set. It printed the mismatch, printed the warning,
# and exited 0 -- a gate that reports and cannot fail, which is worse than no gate because the
# output looks like it was checked. Caught by mutating the manifest and reading the EXIT CODE
# rather than the message.
# ---- the JLL manifest has to name the version its Project.toml claims ---------------------------
#
# `ferrotherm_jll/Project.toml` carries a version and `Artifacts.toml` carries the URLs the binaries
# are actually fetched from, and NOTHING compared them. They drifted at v0.18.0: the release job
# built and uploaded all three platforms, then failed to commit the manifest after a rebase conflict
# -- so Project.toml said 0.18.0 while every download URL still pointed at the v0.17.0 tarballs.
#
# That is the worst shape a version bug takes here. `Pkg.add` resolves, `using Ferrotherm` succeeds,
# the library LOADS -- and it is the previous release's library, answering with the previous
# release's behaviour under this release's version number. Nothing breaks; it is just wrong.
jll_toml="julia/ferrotherm_jll/Project.toml"
jll_art="julia/ferrotherm_jll/Artifacts.toml"
if [[ -f "$jll_toml" && -f "$jll_art" ]]; then
  jll_v="$(grep -m1 '^version' "$jll_toml" | cut -d'"' -f2)"
  art_vs="$(grep -oE 'releases/download/v[0-9]+\.[0-9]+\.[0-9]+' "$jll_art" | sed 's|.*/v||' | sort -u | tr '\n' ' ')"
  art_vs="${art_vs% }"
  if [[ -z "$art_vs" ]]; then
    printf '  %-36s %s\n' "$jll_art" "no download URLs -- the JLL would fetch nothing"
    bad=1
  elif [[ "$art_vs" == "$jll_v" ]]; then
    printf '  %-36s %s\n' "$jll_art" "artifacts point at v$jll_v"
  else
    # BEHIND IS ONLY WRONG ONCE THE BINARIES EXIST.
    #
    # Between bumping the version and the release job finishing there is a legitimate window where
    # Project.toml is ahead of the manifest -- the tarballs are not built yet, so there is nothing
    # for it to point at. Failing there would make every release red on the way out, which is how a
    # gate gets ignored. The same allowance the crates.io rows above make for an unpushed tree.
    #
    # What is NOT legitimate is the release having shipped binaries the manifest never picked up.
    # That is what happened at v0.18.0: all three tarballs uploaded, the manifest commit lost a
    # rebase, and the JLL went on serving v0.17.0 under a 0.18.0 version number.
    if ! command -v gh >/dev/null 2>&1; then
      printf '  %-36s %s\n' "$jll_art" "points at [$art_vs], not v$jll_v -- publish state NOT checked (no gh)"
    elif ! gh release view "v$jll_v" --json assets >/dev/null 2>&1; then
      printf '  %-36s %s\n' "$jll_art" "points at [$art_vs]; no v$jll_v release yet, so nothing to point at"
    elif [[ "$(gh release view "v$jll_v" --json assets --jq '[.assets[] | select(.name|endswith(".tar.gz"))] | length' 2>/dev/null || echo 0)" -eq 0 ]]; then
      printf '  %-36s %s\n' "$jll_art" "points at [$art_vs]; v$jll_v release has no tarballs yet"
    else
      printf '  %-36s artifacts point at [%s], Project.toml says %s\n' "$jll_art" "$art_vs" "$jll_v"
      echo "the v$jll_v release HAS tarballs and the manifest never picked them up." >&2
      echo "the JLL would resolve, load, and hand back the WRONG library." >&2
      echo "  gh release download v$jll_v -D <dir> && julia scripts/rebuild-julia-manifest.jl $jll_v <dir> $jll_art" >&2
      bad=1
    fi
  fi
fi

# The server versions independently -- it is a separate crate with its own release cadence -- but
# the ferrotherm it depends on must be the one in this repository, or `cargo publish` resolves to
# whatever is already on crates.io and quietly ships against an older library.
# Every workspace member that pins the library, FOUND rather than listed. This block named only
# serve, so bumping to 0.9.0 left cloud and silicon pinned at 0.8 and this script said "all agree"
# -- the build was what caught it. A list of places to check is a list that goes stale the moment
# someone adds a crate.
major_minor="${want%.*}"
found=0
while IFS= read -r f; do
  dep="$(grep -m1 '^ferrotherm = ' "$f" | sed -E 's/.*version = "([^"]+)".*/\1/')"
  [[ -n "$dep" ]] || continue
  found=$((found + 1))
  if [[ "$dep" == "$major_minor" || "$dep" == "$want" ]]; then
    printf '  %-36s depends on ferrotherm %s\n' "$f" "$dep"
  else
    printf '  %-36s depends on ferrotherm %s   <- expected %s\n' "$f" "$dep" "$major_minor"
    bad=1
  fi
done < <(owned_manifests | xargs grep -l '^ferrotherm = ' 2>/dev/null | sort)

if [[ $found -eq 0 ]]; then
  # A floor: if the search stops matching, this passes vacuously over nothing.
  echo "found no crate depending on ferrotherm, which cannot be right" >&2
  exit 2
fi

# ---- and is any of it actually ON crates.io? ---------------------------------------------------
#
# Everything above compares the repository against itself, which is why it was green the whole time
# `ferrotherm-gpu` sat at 0.2.0 in the tree and 0.1.0 on crates.io. The bump was committed, tagged,
# described in the changelog and pushed; `cargo add ferrotherm-gpu` still gave you 0.1.0 and a
# `Gpu` with no `is_hardware`. Nothing was inconsistent -- the repository agreed with itself
# perfectly. It just was not shipped, and no check here could see that, because none of them looked
# outside the directory.
#
# The comment at line 55 already reasoned about crates.io. Reasoning about a registry in prose is
# not querying it. That gap is the whole lesson: a failure mode named in a comment and checked by
# nothing is a failure mode this project ships.
#
# Ahead-of-registry is NOT by itself wrong -- between the bump commit and `cargo publish` every
# crate is ahead, and a check that fires there is a check people learn to ignore. The defect is
# ahead AND already pushed to main: at that point the release is announced and the artifact is not
# there. So the condition is deliberately narrow.
#
# The sparse index rather than the JSON API: the API refuses requests without a User-Agent and
# answers a bare curl with something that parses as "no such crate", which would have made this
# report every crate unpublished and be believed exactly once.
index_url() {  # crates.io sparse-index path convention
  local n="$1"
  case ${#n} in
    1) echo "https://index.crates.io/1/$n" ;;
    2) echo "https://index.crates.io/2/$n" ;;
    3) echo "https://index.crates.io/3/${n:0:1}/$n" ;;
    *) echo "https://index.crates.io/${n:0:2}/${n:2:2}/$n" ;;
  esac
}

echo
# "Is this the released state?" -- asked in a way that survives a shallow checkout.
#
# The obvious form is `HEAD == origin/main`, and in CI that quietly answers no: actions/checkout
# defaults to fetch-depth 1, so `origin/main` need not exist as a remote-tracking ref at all, the
# comparison fails, and the block downgrades itself to never-fails on the one runner where it
# matters most. So CI-on-main is asked of the environment directly, and the git comparison is the
# local fallback.
pushed=0
if [[ "${GITHUB_EVENT_NAME:-}" == "push" && "${GITHUB_REF:-}" == "refs/heads/main" ]]; then
  pushed=1
elif git rev-parse --verify -q origin/main >/dev/null 2>&1 \
   && [[ "$(git rev-parse HEAD)" == "$(git rev-parse origin/main)" ]] \
   && [[ -z "$(git status --porcelain)" ]]; then pushed=1; fi

unshipped=0; ahead=0; checked=0; offline=0
while IFS= read -r f; do
  name="$(grep -m1 '^name = ' "$f" | cut -d'"' -f2)"
  [[ -n "$name" ]] || continue
  grep -q '^publish = false' "$f" && continue
  local_v="$(grep -m1 '^version = ' "$f" | cut -d'"' -f2)"

  # Branch on the HTTP STATUS, not on whether curl succeeded.
  #
  # The first cut used `curl -fsS ... || offline=1`, and a never-published crate answers 404, which
  # `-f` reports as failure. So `ferrotherm-cloud` -- genuinely unpublished -- read as "the network
  # is down", broke the loop, and the run announced "publish state not checked" before ever reaching
  # `ferrotherm-gpu`, the one crate this block was written to catch. A gate that cannot tell
  # "absent" from "could not look" reports the reassuring one.
  resp="$(curl -sS --max-time 20 -w '\n%{http_code}' "$(index_url "$name")" 2>/dev/null)"
  code="$(printf '%s' "$resp" | tail -1)"
  body="$(printf '%s' "$resp" | sed '$d')"
  case "$code" in
    200) ;;
    404) body="" ;;
    *)   offline=1; break ;;
  esac
  # Every line is one version; yanked ones do not count as published.
  live="$(printf '%s\n' "$body" \
      | python3 -c 'import json,sys; vs=[json.loads(l)["vers"] for l in sys.stdin if l.strip() and not json.loads(l).get("yanked")]; print(vs[-1] if vs else "")' 2>/dev/null)"
  checked=$((checked + 1))

  if [[ -z "$live" ]]; then
    printf '  %-24s %-9s crates.io: never published\n' "$name" "$local_v"
    ahead=$((ahead + 1)); [[ $pushed -eq 1 ]] && unshipped=$((unshipped + 1))
  elif [[ "$live" == "$local_v" ]]; then
    printf '  %-24s %-9s crates.io: %s\n' "$name" "$local_v" "$live"
  elif [[ "$(printf '%s\n%s\n' "$live" "$local_v" | sort -V | tail -1)" == "$local_v" ]]; then
    printf '  %-24s %-9s crates.io: %-9s <- ahead\n' "$name" "$local_v" "$live"
    ahead=$((ahead + 1)); [[ $pushed -eq 1 ]] && unshipped=$((unshipped + 1))
  else
    # Behind the registry: someone published from elsewhere, or a bump was reverted. Not this
    # check's failure to raise, but say it rather than swallow it.
    printf '  %-24s %-9s crates.io: %-9s <- registry is AHEAD of this tree\n' "$name" "$local_v" "$live"
  fi
done < <(owned_manifests | sort)

if [[ $offline -eq 1 ]]; then
  echo "  (crates.io unreachable -- publish state not checked)"
elif [[ $checked -eq 0 ]]; then
  echo "found no crate to look up, which cannot be right" >&2
  exit 2
elif [[ $unshipped -gt 0 ]]; then
  echo
  echo "$unshipped crate(s) are newer here than on crates.io, on a commit already pushed to main." >&2
  echo "The changelog says shipped; \`cargo add\` disagrees." >&2
  echo >&2
  echo "  cargo publish -p <crate>          to ship it, or" >&2
  echo "  publish = false in its Cargo.toml to say it is internal on purpose" >&2
  echo >&2
  echo "A crate carrying a description, licence, keywords and categories was written to be" >&2
  echo "published. Leaving it unpublished and unmarked is the state that is neither." >&2
  bad=1
fi

if [[ $bad -eq 1 ]]; then
  echo
  echo "these are released together; a version that disagrees is a version nobody can use." >&2
  exit 1
fi
# Say what was actually checked. The first cut printed "every crate is on crates.io at its tree
# version" after looking up one crate and giving up on the rest, which is a success line asserting
# something the run had not established.
if [[ $offline -eq 1 ]]; then
  echo "all agree on $want (publish state NOT checked -- crates.io was unreachable)"
elif [[ $ahead -gt 0 ]]; then
  # Ahead but not failing, because the tree is dirty or unpushed -- mid-release, which is fine.
  # Say so anyway. The first cut printed the all-clear here while the rows above it showed two
  # crates that were not on crates.io at all: a summary that contradicts its own table trains
  # people to read the summary and skip the table.
  echo "all agree on $want; $ahead crate(s) above are ahead of crates.io -- not failing, because"
  echo "this tree is dirty or unpushed. They must be published before the release is real."
else
  echo "all agree on $want, and all $checked crates are on crates.io at their tree version"
fi
