#!/usr/bin/env bash
#
# Cut a release: verify, tag, push, and publish every crate FROM A PRISTINE WORKTREE.
#
#   scripts/cut-release.sh <version>              # full cut: verify tree state, tag, push, publish
#   scripts/cut-release.sh <version> --publish-only   # tag exists; (re)do the publishes
#
# WHY THE WORKTREE, with the incident. At the 0.36.0 cut, `cargo publish` refused because another
# session's uncommitted edit sat in src/. The workaround was `git stash push <path>` / publish /
# `git stash pop` -- three commands, by hand, against a live shared tree, where forgetting the pop
# loses someone's work and `--allow-dirty` (cargo's own suggestion) would have shipped unreviewed
# bytes to a registry that keeps them forever. This repo's trees are SHARED between sessions by
# design, so "keep the tree clean during a cut" is not a rule anyone can actually promise.
#
# So the publish does not read the tree at all. `git worktree add --detach <tmp> v<version>` gives
# the tagged commit and nothing else; the six crates publish from there in dependency order; the
# worktree is removed on exit either way. A dirty main tree, another session mid-edit, scratch
# directories -- none of it can reach the registry, and nothing needs stashing.
#
# WHAT THIS REFUSES, and why each refusal is a fact and not a formality:
#   * a version that does not match Cargo.toml at the tag -- the tag names the wrong commit;
#   * a CHANGELOG at the tag still carrying "## Unreleased" -- the cut commit was not made;
#   * an un-pushed tag -- crates.io must never be ahead of the public history it claims to be.
#
# The publish order is the dependency order: core, then meter (gpu needs it), then the rest.
# A crate already on the registry at its version is SKIPPED with a word, not an error, so a cut
# interrupted halfway resumes by re-running (the 0.36.0 cut published core and then hit the dirty
# tree five times; this rerun-safety is that afternoon, encoded).

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
version="${1:-}"
mode="${2:-}"
[[ -n "$version" ]] || { echo "usage: cut-release.sh <version> [--publish-only]" >&2; exit 2; }
tag="v$version"

cd "$here"

# ---- the tag must exist, be pushed, and agree with itself --------------------------------------
git rev-parse -q --verify "refs/tags/$tag" >/dev/null || {
    echo "no tag $tag. Make the release commit (versions + CHANGELOG), tag it, then rerun." >&2
    exit 1
}
tagged_version=$(git show "$tag:Cargo.toml" | grep -m1 '^version' | cut -d'"' -f2)
[[ "$tagged_version" == "$version" ]] || {
    echo "tag $tag has Cargo.toml version $tagged_version -- the tag names the wrong commit." >&2
    exit 1
}
if git show "$tag:CHANGELOG.md" | grep -q '^## Unreleased'; then
    echo "CHANGELOG at $tag still says '## Unreleased' -- the release commit was not finished." >&2
    exit 1
fi
if ! git ls-remote --tags origin "refs/tags/$tag" | grep -q .; then
    echo "tag $tag is not on origin. Push it first: the registry must never lead the history." >&2
    exit 1
fi

# ---- and MAIN must not be there yet, or CI is already racing these uploads -----------------------
#
# This is a warning and not a refusal, because by the time you can read it the push has happened and
# refusing would leave the release half-made. It exists because the 0.13.0 incident recurred at
# 0.40.0 for a reason nobody had written down: RELEASING.md said "push last", cut-release.sh requires
# the TAG pushed first, and the obvious reading -- push both, then publish -- puts CI on main inside
# the upload window. It ran while ferrotherm-silicon was still going up and failed on one crate
# reading "<- ahead", with nothing wrong but the order. The order is: tag, publish, THEN main.
# Ancestry, not equality: by the time anyone reads this, origin/main has usually moved on (the
# release workflow commits the JLL manifest back), and comparing tip hashes would silently never fire
# -- a warning that cannot trigger is worse than none, because it reads as "checked, fine".
main_has_tag=0
if [[ "$mode" != "--publish-only" ]]; then
    git fetch -q origin main 2>/dev/null || true
    git merge-base --is-ancestor "$tag^{commit}" FETCH_HEAD 2>/dev/null && main_has_tag=1
fi
if [[ "$main_has_tag" -eq 1 ]]; then
    echo
    echo "  NOTE: origin/main already has $tag's commit, so CI is running against a registry that"
    echo "  does not have these versions yet. check-versions.sh will correctly report the crates"
    echo "  still uploading as '<- ahead' and that job will go red. It is a race, not a defect:"
    echo "  when this script reports all six live, re-run the failed job."
    echo "  Next time: push the tag, run this, and push main LAST (see RELEASING.md)."
    echo
fi

# ---- publish from a worktree that contains the tag and nothing else ----------------------------
wt=$(mktemp -d "${TMPDIR:-/tmp}/ferrotherm-cut-XXXXXX")
cleanup() { git worktree remove --force "$wt" >/dev/null 2>&1 || true; rm -rf "$wt"; }
trap cleanup EXIT
git worktree add --detach "$wt" "$tag" >/dev/null

# Crate name -> the version it carries at the tag, in dependency order.
crates=(ferrotherm ferrotherm-meter ferrotherm-gpu ferrotherm-cloud ferrotherm-serve ferrotherm-silicon)
published=0 skipped=0
for crate in "${crates[@]}"; do
    dir="."
    case "$crate" in
        ferrotherm-meter) dir=meter ;; ferrotherm-gpu) dir=gpu ;;
        ferrotherm-cloud) dir=cloud ;; ferrotherm-serve) dir=serve ;;
        ferrotherm-silicon) dir=silicon ;;
    esac
    cver=$(grep -m1 '^version' "$wt/$dir/Cargo.toml" | cut -d'"' -f2)
    # Already on the registry at this version? Then this cut (or a previous run of it) did the job.
    # crates.io refuses requests with no User-Agent, so name ourselves -- an anonymous probe here
    # reads as "not published" and turns every skip into a doomed publish attempt.
    if curl -fsSL -A "ferrotherm-cut-release (dcharlot@ucsb.edu)" \
            "https://crates.io/api/v1/crates/$crate/$cver" 2>/dev/null | grep -q '"num"'; then
        echo "  $crate $cver already on crates.io -- skipped"
        skipped=$((skipped + 1))
        continue
    fi
    echo "  publishing $crate $cver from the $tag worktree..."
    (cd "$wt" && cargo publish -p "$crate" --quiet)
    published=$((published + 1))
done

echo "published $published crate(s), $skipped already present; verifying the registry agrees..."
FERROTHERM_SKIP_JLL_DRIFT="${FERROTHERM_SKIP_JLL_DRIFT:-}" bash scripts/check-versions.sh
