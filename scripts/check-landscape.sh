#!/usr/bin/env bash
#
# Has the competitive landscape moved enough to be worth re-surveying?
#
# The 2026-08-05 survey (11 agents, 687 tool calls, primary sources in five languages) is the
# expensive thing. Re-running it is a day's work, and the question that decides whether to is
# cheap — so this asks the cheap question, and the expensive one stays parked until the answer
# changes.
#
# Each block below is a FALSIFIABLE claim that survey staked. Not an impression, not a vibe: a
# statement about a file, a repository or a metric that a command can check. If they all still hold,
# the map is current and re-surveying would rediscover it. If one breaks, that specific claim is what
# moved, and it is the thing to go and look at.
#
#   scripts/check-landscape.sh              re-check the claims
#   scripts/check-landscape.sh --selftest   prove the claim checks can actually fire
#
# Needs `gh` authenticated. Without network it says so and exits 2 rather than reporting "unchanged",
# because "I could not look" and "nothing moved" are different answers and only one of them is safe.

set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

command -v gh >/dev/null 2>&1 || { echo "needs the gh CLI, authenticated" >&2; exit 2; }
gh api rate_limit >/dev/null 2>&1 || {
  echo "cannot reach the GitHub API: this run checked NOTHING" >&2
  exit 2
}

moved=0
say()  { printf '  %-52s %s\n' "$1" "$2"; }
broke() { printf '  %-52s ** %s **\n' "$1" "$2"; moved=$((moved + 1)); }

# ---- the two patterns every claim below is decided by ------------------------------------------
#
# Hoisted out of the claim blocks so that --selftest runs the SAME patterns against a damaged copy
# of a real tree. A selftest that exercises its own private copy of the rule proves nothing about
# the rule the gate uses; there is one definition of each and both modes read it here.
#
# DEVICE_RX deliberately excludes `xtr` on its own: it matches `extropic` in every brand asset,
# which is how a first cut of this check reported a wordmark PNG as a device driver.
DEVICE_RX='(^|/)(device|driver|hardware)[^/]*\.(py|rs|c|h|cpp)$|pcie|/xtr[0-9_-]|usb|ioctl'
# NOT bare "energy". In this field that word almost always means the ISING energy -- the value
# of the objective function -- and matching it flagged `docs/screenshots/target_energy.png`, a
# picture of a solver's cost curve, as though a competitor had started reporting power. The two
# senses of the word are the whole distinction this project is built on, so the pattern has to
# know the difference: joules, watts and consumption are electrical; "energy" alone is not.
ENERGY_RX='joule|watt|kwh|power.?(consum|draw|meter)|energy.?(consum|per.?op|per.?sample|efficien)'

# The remote artefact each claim is decided on, and the one comparison that decides it. Both modes
# go through these two functions, so neutering the comparison neuters the selftest with it.
tree_paths() { gh api "repos/$1/git/trees/$2?recursive=1" --jq '.tree[].path' 2>/dev/null; }
count_hits() { grep -icE "$1"; }   # paths on stdin, count of matching lines on stdout

# ---- --selftest: prove a claim check can fail ---------------------------------------------------
#
# WHY THIS DAMAGE. Both claims are decided the same way: fetch a repository's file listing and
# require a pattern to match NOTHING. That shape rots silently in both directions, and neither
# direction shows up in a green run:
#
#   * the pattern stops matching — a tightening to kill a false positive that went one clause too
#     far, or a listing that arrives empty because the API changed shape — and then "none — claim
#     holds" means "I looked at nothing", which is the one answer this gate exists to refuse;
#   * the pattern matches everything — a widening to catch a near-miss — and the gate cries wolf
#     until nobody reads it, at which point the real hit is ignored too.
#
# So each case does two things to one fetched listing. It APPENDS the regression, in the form it
# would actually arrive in — a device backend module landing in torx is what "native Z1 execution"
# ships as, and `energy_per_sample` is what an energy axis appearing in a competitor's benchmark
# looks like as a filename — and it requires the pattern to flag it. And it names a NEAR-MISS that
# is already in that same listing, upstream, today: `extropic_wordmark.png` and the screenshot
# `target_energy.png`, the two paths that each fooled a first cut of this check. The near-miss is
# not injected, it is asserted present and required to stay unflagged, so a widened pattern is
# caught against real upstream files rather than against one I invented to be easy.
#
# Nothing is written outside a mktemp dir; the trees are read-only fetches of somebody else's repo.
if [[ "${1:-}" == "--selftest" ]]; then
  tmp=$(mktemp -d) || { echo "SELFTEST FAILED: no scratch dir, so nothing could be damaged" >&2; exit 2; }
  trap 'rm -rf "$tmp"' EXIT

  # repo|which pattern|the regression to append|a real upstream path that must stay unflagged
  # bash 3.2 has no associative arrays, so: a pipe-delimited table read in the current shell.
  while IFS='|' read -r repo which inject nearmiss; do
    [[ -n "$repo" ]] || continue
    case "$which" in
      device) rx="$DEVICE_RX" ;;
      energy) rx="$ENERGY_RX" ;;
      *)      echo "SELFTEST FAILED: no pattern is named $which" >&2; exit 2 ;;
    esac

    branch=$(gh api "repos/$repo" --jq .default_branch 2>/dev/null)
    if [[ -z "$branch" ]]; then
      echo "SELFTEST FAILED: $repo is unreachable, so no tree was fetched, so nothing was damaged" >&2
      echo "                 and nothing was proved. That is the could-not-look answer, not a pass." >&2
      exit 2
    fi
    tree_paths "$repo" "$branch" > "$tmp/real.txt"
    if [[ ! -s "$tmp/real.txt" ]]; then
      echo "SELFTEST FAILED: the listing for $repo came back EMPTY, so there was nothing to damage" >&2
      echo "                 -- and an empty listing is exactly what the live check reads as clean." >&2
      exit 2
    fi

    # The near-miss has to still be up there, or nothing in this run demonstrates that the pattern
    # can tell a brand asset or an Ising-energy screenshot from a regression. Upstream is free to
    # move its own files, and when it does this line is the thing to update -- to another near-miss,
    # not to nothing.
    if ! grep -qxF "$nearmiss" "$tmp/real.txt"; then
      echo "SELFTEST FAILED: $repo no longer contains $nearmiss, so this run did not show that the" >&2
      echo "                 pattern can tell a near-miss from a regression. Point this case at a" >&2
      echo "                 near-miss that is still there." >&2
      exit 1
    fi

    # The claim must hold on the UNDAMAGED copy, or a hit on the damaged one says nothing about the
    # damage -- and since the near-miss is in this listing, a zero here IS the proof that the
    # pattern ignores it. Two very different things trip this and the reader has to tell them apart:
    # either the claim genuinely moved (go look, the survey is due), or the pattern was widened
    # until it matches ordinary upstream files, which is the false-alarm rot described above.
    base=$(count_hits "$rx" < "$tmp/real.txt")
    if [[ "${base:-0}" -ne 0 ]]; then
      echo "SELFTEST FAILED: $repo already matches this pattern undamaged (${base:-0} path(s)), so a" >&2
      echo "                 hit on the damaged copy would prove nothing. Either the claim moved --" >&2
      echo "                 run scripts/check-landscape.sh with no arguments and read what it names" >&2
      echo "                 -- or the pattern has been widened until it matches ordinary files." >&2
      exit 1
    fi

    cp "$tmp/real.txt" "$tmp/damaged.txt"
    printf '%s\n' "$inject" >> "$tmp/damaged.txt"
    # Prove the damage LANDED. A comparison that "fails" because the injection never reached the
    # copy is a vacuous selftest wearing a green tick. Two ways of asking, because each catches a
    # different way of not landing: the line is there, and the copy is one line longer than the
    # listing it came from.
    if ! grep -qxF "$inject" "$tmp/damaged.txt"; then
      echo "SELFTEST FAILED: the damage did not apply -- $inject is not in the damaged copy" >&2
      exit 1
    fi
    if [[ "$(wc -l < "$tmp/damaged.txt")" -ne "$(( $(wc -l < "$tmp/real.txt") + 1 ))" ]]; then
      echo "SELFTEST FAILED: the damage did not apply -- the damaged copy of $repo is not exactly" >&2
      echo "                 one path longer than the listing it was copied from" >&2
      exit 1
    fi

    hits=$(count_hits "$rx" < "$tmp/damaged.txt")
    if [[ "${hits:-0}" -eq 0 ]]; then
      echo "SELFTEST FAILED: $repo -- $inject went through the check unflagged, so this claim would" >&2
      echo "                 still read \"none, claim holds\" after the thing it watches for landed." >&2
      exit 1
    fi
    # Exactly one, because the listing scored zero a moment ago and one path was added. More than
    # one means the two counts disagree about the same file list, which is not a landscape question
    # at all -- it is this script losing track of what it is grepping.
    if [[ "${hits:-0}" -ne 1 ]]; then
      echo "SELFTEST FAILED: $repo -- the damaged listing scored ${hits} where the undamaged one" >&2
      echo "                 scored 0 and exactly one path was added; the two counts are not looking" >&2
      echo "                 at the same list." >&2
      exit 1
    fi
    printf '  selftest  %-27s flags %-44s and still ignores %s\n' "$repo" "$inject" "$nearmiss"
  done <<'CASES'
extropic-ai/torx|device|src/torx/backend/device_z1.py|docs_site/assets/brand/extropic_wordmark.png
fixstars/amplify-benchmark|energy|amplify_bench/metrics/energy_per_sample.py|docs/screenshots/target_energy.png
CASES

  echo "selftest ok: both claim patterns fire on the regression they watch for, and neither fires on"
  echo "the near-miss that is sitting in the same listing and fooled an earlier cut of this check."
  exit 0
fi

echo "Claims from the 2026-08-05 survey, re-checked:"
echo

# ---- CLAIM 1: no public path from any library to any silicon, anywhere ------------------------
#
# The load-bearing one. Extropic markets TORX as running on XTR-0 today, and their own torx paper
# lists native Z1 execution as OPEN WORK; every published "backend" is a simulator class. If device
# code ever lands in these repos, the empty lane this project occupies has an occupant.
#
# sparse-transformers joined the watch on 2026-09-05, the day after the Z1T release. That release is
# what "models for Z1" ships as, and it is the repo native execution would arrive in -- so far it is
# JAX training code that runs on GPUs, and the energy figures beside it come from a device model in
# a blog post rather than from anything in the tree. That gap is the claim.
for repo in extropic-ai/thrml extropic-ai/torx extropic-ai/sparse-transformers; do
  branch=$(gh api "repos/$repo" --jq .default_branch 2>/dev/null)
  if [[ -z "$branch" ]]; then
    say "$repo" "unreachable (repo moved or private?)"
    continue
  fi
  hits=$(tree_paths "$repo" "$branch" | count_hits "$DEVICE_RX")
  if [[ "$hits" -eq 0 ]]; then
    say "$repo: device code" "none — claim holds"
  else
    broke "$repo: device code" "$hits path(s) — GO LOOK"
  fi
done

# ---- CLAIM 2: nobody reports joules -----------------------------------------------------------
#
# amplify-benchmark's four metrics have no energy axis; Normal's CN101 paper reports zero watts
# while marketing "up to 1000x". An energy axis appearing in a competitor's benchmark is the single
# development that would most change this project's positioning, because measuring joules is the
# lane it is built in.
#
# REFINED 2026-09-05 by the Z1T release, which DOES report joules: 294.52 nJ/token split 8.74 Z1 /
# 285.78 FPGA, against an H100 measured at 40.9 uJ/token. So the prose claim "nobody reports joules"
# no longer holds as written, and the surviving claim is narrower and still true: nobody reports
# MEASURED joules for a thermodynamic sampler. Z1 is at tapeout and its term is projected from X0
# pbits; the only measured number in that ratio is the GPU it is divided by. The arithmetic of that
# release is reproduced in `examples/z1t_ledger.rs` -- including the part the headline does not
# contain, that zeroing the sampler entirely moves 138.9x to 143.1x.
#
# What this gate watches for is therefore unchanged: a joules axis inside a competitor's BENCHMARK,
# where a number has to be produced by running something.
for repo in fixstars/amplify-benchmark; do
  branch=$(gh api "repos/$repo" --jq .default_branch 2>/dev/null)
  hits=$(tree_paths "$repo" "$branch" | count_hits "$ENERGY_RX")
  if [[ "$hits" -eq 0 ]]; then
    say "$repo: an energy metric" "none — claim holds"
  else
    broke "$repo: an energy metric" "$hits path(s) — GO LOOK"
  fi
done

# ---- CLAIM 3: the field is quiet ---------------------------------------------------------------
#
# Staleness is the cheapest possible proxy and it is a real one: a survey is worth re-running when
# the things it surveyed have changed, and a repository nobody has pushed to in months has not.
# Reported rather than judged — the numbers are the output, and a human decides what they mean.
echo
echo "Last push, as a staleness proxy:"
for repo in extropic-ai/thrml extropic-ai/torx extropic-ai/sparse-transformers \
            normal-computing/thermox OpenJij/OpenJij \
            dwavesystems/dwave-ocean-sdk fixstars/amplify-benchmark; do
  info=$(gh api "repos/$repo" --jq '"\(.pushed_at[0:10])|\(.stargazers_count)"' 2>/dev/null)
  if [[ -z "$info" ]]; then
    printf '  %-38s %s\n' "$repo" "unreachable"
    continue
  fi
  printf '  %-38s %s  (%s stars)\n' "$repo" "${info%%|*}" "${info##*|}"
done

echo
if [[ $moved -gt 0 ]]; then
  echo "$moved claim(s) from the survey no longer hold. That is the trigger to re-walk — and the" >&2
  echo "broken claim is where to start, rather than the whole map." >&2
  exit 1
fi
echo "  every checked claim still holds; the 2026-08-05 map is current."
echo "  Re-surveying now would rediscover it. Run this again rather than guessing."
