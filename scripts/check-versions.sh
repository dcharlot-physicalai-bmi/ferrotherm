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
#   scripts/check-versions.sh              every manifest, every pin and crates.io agree
#   scripts/check-versions.sh --selftest   prove the comparison can fail

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

# ---- --selftest: prove this gate can fail ------------------------------------------------------
#
# This gate spent its whole life green, and it was found reporting "all 54 crates are on crates.io
# at their tree version" for a repository that has SIX -- it had been enumerating through eight
# nested worktrees, inflating its own coverage 9x, and calling a correct release a regression
# seconds after it happened. Nothing had ever demonstrated it could fail, so nobody looked. A gate
# nobody has proved can fail is not a gate.
#
# So: damage a COPY of the tree in the two shapes this gate exists to catch, and require it to
# object to each one, by name.
#
#   1. ONE SIBLING MANIFEST OUT OF STEP. The library, the wheel and the two Julia manifests are
#      released together and bumped by hand in five places. Forgetting one -- or bumping one alone
#      -- is the likeliest mistake anyone makes in this repository, and it ships a Python package
#      whose version number describes a library it does not contain.
#   2. A DEPENDENCY PIN LEFT AT THE PREVIOUS MINOR. This has already happened: bumping the library
#      to 0.9.0 left cloud and silicon pinned at 0.8, and `cargo publish` then resolves the pin
#      against crates.io and quietly ships a server built on the OLD library. The comment above the
#      pin loop is the account of it. Nothing about that state is a build error.
#
# Neither is a strawman -- both are edits a person makes in one keystroke, and both leave every file
# well-formed, parseable and plausible. Truncating a file is not damage, it is vandalism, and
# everything catches it.
#
# The copy is a real git checkout (`git init` + `git add`), not a loose directory, because
# `owned_manifests` enumerates through `git ls-files` and the selftest must exercise the SAME
# enumeration the gate uses in anger -- the non-git fallback is not the code under test, and
# enumeration is where this gate's own bug was.
#
# The undamaged copy is run FIRST and required to PASS. Without that control, a damaged run that
# fails for an environmental reason -- a file the copy forgot, a broken interpreter -- reads as a
# caught regression, which is exactly the self-congratulating green this mode exists to refuse.
#
# NOT COVERED, and worth saying rather than implying otherwise: the crates.io rows and the JLL
# Artifacts.toml drift check both need state outside the tree (a live registry, and `gh` able to see
# a real release carrying tarballs). A temp checkout has no remote, so those two arms cannot be
# damaged into a deterministic failure here. This proves the local agreement comparisons can fail;
# it does not prove the registry ones can.
if [[ "${1:-}" == "--selftest" ]]; then
  ver="$(read_version Cargo.toml)"
  if [[ -z "$ver" ]]; then
    echo "SELFTEST FAILED: could not read a version out of Cargo.toml to damage" >&2
    exit 1
  fi

  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT

  # Everything this gate reads, and nothing else. `${files[@]}` is reused rather than retyped, so a
  # manifest added to the list above is automatically part of the selftest instead of quietly
  # outside it.
  manifests="$(owned_manifests | sed 's|^\./||' | sort)"
  copy_list="$(printf '%s\n' "${files[@]}" julia/ferrotherm_jll/Artifacts.toml \
                              scripts/check-versions.sh "$manifests")"
  while IFS= read -r f; do
    [[ -n "$f" && -f "$f" ]] || continue
    mkdir -p "$tmp/$(dirname "$f")"
    cp "$f" "$tmp/$f"
  done <<< "$copy_list"

  if ! git -C "$tmp" init -q . >/dev/null 2>&1 || ! git -C "$tmp" add -A >/dev/null 2>&1; then
    echo "SELFTEST FAILED: could not make the copy a git checkout, so the enumeration under test" >&2
    echo "would not be the one this gate actually uses" >&2
    exit 1
  fi

  run_copy() { ( bash "$tmp/scripts/check-versions.sh" ) 2>&1; }

  if ! control="$(run_copy)"; then
    echo "SELFTEST FAILED: the UNDAMAGED copy did not pass, so a failure on a damaged one would" >&2
    echo "prove nothing about the comparison. The copy said:" >&2
    printf '%s\n' "$control" >&2
    exit 1
  fi

  # How many crates the CONTROL run counted, read out of its own success line rather than assumed.
  # Damage 3 requires this number not to move; hardcoding six would stop being true the day a
  # seventh crate lands, and a selftest that goes stale is the thing this file is arguing against.
  n_owned="$(printf '%s\n' "$control" | sed -n 's/.*and all \([0-9][0-9]*\) crates.*/\1/p' | head -1)"
  if [[ -z "$n_owned" ]]; then
    echo "SELFTEST FAILED: could not read the crate count out of the control run, so damage 3" >&2
    echo "cannot tell whether the enumeration moved. It said:" >&2
    printf '%s\n' "$control" >&2
    exit 1
  fi

  # Put every copied file back as it was, so one arm's damage cannot leak into the next.
  restore_all() {
    while IFS= read -r f; do
      [[ -n "$f" && -f "$f" ]] || continue
      cp "$f" "$tmp/$f"
    done <<< "$copy_list"
  }

  # --- damage 1: the wheel one patch ahead of the crate -------------------------------------------
  bumped="${ver%.*}.$(( ${ver##*.} + 1 ))"
  py="$tmp/python/ferrotherm/__init__.py"
  sed "s|^__version__ = \"$ver\"|__version__ = \"$bumped\"|" "$py" > "$py.tmp"
  mv "$py.tmp" "$py"
  if ! grep -q "^__version__ = \"$bumped\"" "$py"; then
    echo "SELFTEST FAILED: the damage did not apply -- __version__ in the copied wheel is not" >&2
    echo "$bumped, so the run below would have been vacuous" >&2
    exit 1
  fi
  if out="$(run_copy)"; then
    echo "SELFTEST FAILED: the gate passed a tree whose crate says $ver and whose Python package" >&2
    echo "says $bumped -- exactly the drift it exists to catch" >&2
    exit 1
  fi
  case "$out" in
    *"python/ferrotherm/__init__.py"*"expected $ver"*) ;;
    *) echo "SELFTEST FAILED: the damaged run failed, but not on the wheel version -- the failure" >&2
       echo "came from somewhere else, so that comparison is still unproven. It said:" >&2
       printf '%s\n' "$out" >&2
       exit 1 ;;
  esac
  cp python/ferrotherm/__init__.py "$py"   # back to pristine before the second damage

  # --- damage 2: a sibling still pinning the previous minor ---------------------------------------
  pin="${ver%.*}"
  pin_maj="${pin%.*}"; pin_min="${pin#*.}"
  # The historical bug is a pin left BEHIND, so go back one minor. On an 0.0.x library there is no
  # minor to go back to; a pin one minor AHEAD is the same disagreement read from the other side,
  # and the gate has to object to it just as loudly.
  if [[ "$pin_min" -gt 0 ]]; then stale="$pin_maj.$((pin_min - 1))"
  elif [[ "$pin_maj" -gt 0 ]]; then stale="$((pin_maj - 1)).0"
  else stale="$pin_maj.$((pin_min + 1))"; fi

  dep_file=""
  while IFS= read -r f; do
    [[ -n "$f" && -f "$tmp/$f" ]] || continue
    if grep -q '^ferrotherm = ' "$tmp/$f"; then dep_file="$f"; break; fi
  done <<< "$manifests"
  if [[ -z "$dep_file" ]]; then
    echo "SELFTEST FAILED: no copied manifest pins ferrotherm, so there was nothing to damage" >&2
    exit 1
  fi

  sed "/^ferrotherm = /s|version = \"$pin\"|version = \"$stale\"|" "$tmp/$dep_file" > "$tmp/$dep_file.tmp"
  mv "$tmp/$dep_file.tmp" "$tmp/$dep_file"
  if ! grep -q "^ferrotherm = .*version = \"$stale\"" "$tmp/$dep_file"; then
    echo "SELFTEST FAILED: the damage did not apply -- $dep_file does not pin ferrotherm $stale," >&2
    echo "so the run below would have been vacuous" >&2
    exit 1
  fi
  if out="$(run_copy)"; then
    echo "SELFTEST FAILED: the gate passed a workspace member pinning ferrotherm $stale against a" >&2
    echo "library at $ver -- that publishes a crate built on the previous release" >&2
    exit 1
  fi
  case "$out" in
    *"depends on ferrotherm $stale"*"expected $pin"*) ;;
    *) echo "SELFTEST FAILED: the damaged run failed, but not on the dependency pin -- the failure" >&2
       echo "came from somewhere else, so that comparison is still unproven. It said:" >&2
       printf '%s\n' "$out" >&2
       exit 1 ;;
  esac

  # --- damage 3: a NESTED CHECKOUT, which is the bug this whole gate mode exists because of ------
  #
  # The two arms above prove the comparisons can fail. Neither proves the ENUMERATION is right, and
  # enumeration is exactly where this gate's own bug was: `find .` descended into git worktrees, so
  # eight of them one commit behind reported all six crates as "registry is AHEAD of this tree"
  # seconds after a correct release, and inflated the coverage line to "all 54 crates" for a
  # repository with six.
  #
  # This arm is INVERTED against the other two. It plants a stale manifest where a worktree would
  # put one and requires the gate to STILL PASS, with its coverage count unmoved.
  #
  # HONESTLY, ON WHICH ARM ACTUALLY CATCHES A REGRESSION HERE. Both `find`-based enumerations tried
  # -- the original CWD-relative one and an anchored `(cd "$here" && find .)` -- are caught by the
  # CONTROL run above rather than by this arm, because a `find` that walks the real tree also
  # misreads the temp copy. So the property the whole mode guarantees is "the selftest goes red when
  # the enumeration regresses", and the control arm is what delivers it today. This arm is still
  # worth its lines: it is the only one that states the requirement POSITIVELY -- a nested checkout
  # must be invisible and must not move the count -- so an enumeration that happened to satisfy the
  # control run and still counted worktrees would fail here and nowhere else.
  restore_all
  nested="$tmp/.claude/worktrees/w"
  mkdir -p "$nested"
  # A whole stale copy of the repository, which is what a worktree actually is.
  while IFS= read -r rel; do
    [[ -n "$rel" ]] || continue
    mkdir -p "$nested/$(dirname "$rel")"
    cp "$tmp/$rel" "$nested/$rel"
  done <<< "$copy_list"
  # ...pinned a minor behind, the way a worktree cut before the release bump is.
  sed "s/^version = \"$ver\"/version = \"$stale.0\"/" "$tmp/Cargo.toml" > "$nested/Cargo.toml.tmp"
  mv "$nested/Cargo.toml.tmp" "$nested/Cargo.toml"
  if ! grep -q "^version = \"$stale.0\"" "$nested/Cargo.toml"; then
    echo "SELFTEST FAILED: the damage did not apply -- the planted nested checkout does not carry" >&2
    echo "a stale version, so this arm proved nothing about enumeration." >&2
    exit 1
  fi
  if ! out="$(cd "$tmp" && "$here/scripts/check-versions.sh" 2>&1)"; then
    echo "SELFTEST FAILED: a stale manifest planted at .claude/worktrees/ made the gate fail. It" >&2
    echo "must be INVISIBLE: a git worktree is not tracked by its parent, and counting one is the" >&2
    echo "bug this mode exists to prevent. The gate said:" >&2
    printf '%s\n' "$out" >&2
    exit 1
  fi
  case "$out" in
    *"all $n_owned crates"*) : ;;
    *) echo "SELFTEST FAILED: the gate passed, but its coverage count changed when a nested" >&2
       echo "checkout was planted -- it is counting manifests it does not own. It said:" >&2
       printf '%s\n' "$out" >&2
       exit 1 ;;
  esac

  echo "selftest: a wheel at $bumped beside a crate at $ver was caught, and so was $dep_file"
  echo "pinning ferrotherm $stale instead of $pin. A stale manifest planted where a git worktree"
  echo "would put one stayed invisible and the coverage count held at $n_owned, so the enumeration"
  echo "this gate was once wrong about is covered as well as its comparisons."
  exit 0
fi

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
