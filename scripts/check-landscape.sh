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
#   scripts/check-landscape.sh
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

echo "Claims from the 2026-08-05 survey, re-checked:"
echo

# ---- CLAIM 1: no public path from any library to any silicon, anywhere ------------------------
#
# The load-bearing one. Extropic markets TORX as running on XTR-0 today, and their own torx paper
# lists native Z1 execution as OPEN WORK; every published "backend" is a simulator class. If device
# code ever lands in these repos, the empty lane this project occupies has an occupant.
#
# The pattern deliberately excludes `xtr` on its own: it matches `extropic` in every brand asset,
# which is how a first cut of this check reported a wordmark PNG as a device driver.
DEVICE_RX='(^|/)(device|driver|hardware)[^/]*\.(py|rs|c|h|cpp)$|pcie|/xtr[0-9_-]|usb|ioctl'
for repo in extropic-ai/thrml extropic-ai/torx; do
  branch=$(gh api "repos/$repo" --jq .default_branch 2>/dev/null)
  if [[ -z "$branch" ]]; then
    say "$repo" "unreachable (repo moved or private?)"
    continue
  fi
  hits=$(gh api "repos/$repo/git/trees/$branch?recursive=1" --jq '.tree[].path' 2>/dev/null \
         | grep -icE "$DEVICE_RX")
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
for repo in fixstars/amplify-benchmark; do
  branch=$(gh api "repos/$repo" --jq .default_branch 2>/dev/null)
  # NOT bare "energy". In this field that word almost always means the ISING energy -- the value
  # of the objective function -- and matching it flagged `docs/screenshots/target_energy.png`, a
  # picture of a solver's cost curve, as though a competitor had started reporting power. The two
  # senses of the word are the whole distinction this project is built on, so the pattern has to
  # know the difference: joules, watts and consumption are electrical; "energy" alone is not.
  hits=$(gh api "repos/$repo/git/trees/$branch?recursive=1" --jq '.tree[].path' 2>/dev/null \
         | grep -icE "joule|watt|kwh|power.?(consum|draw|meter)|energy.?(consum|per.?op|per.?sample|efficien)")
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
for repo in extropic-ai/thrml extropic-ai/torx normal-computing/thermox OpenJij/OpenJij \
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
