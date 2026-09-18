#!/usr/bin/env bash
# Every number this project publishes about ITSELF, re-derived and checked against the documents.
#
# THE HOLE THIS CLOSES. README.md said "699 tests" in one place and "911 tests" in another.
# LANDSCAPE.md said 911. The truth was 1,709. README said "Eighteen external sources of truth" and
# LANDSCAPE said "Twenty-one" — of the same thing, in the same repository, on the same day. An
# outside reader found the drift before we did, and the docs ship to crates.io, so the front page of
# a project whose entire argument is verification was carrying three mutually inconsistent counts.
#
# A number typed into prose rots the moment the suite grows. A number that is derived cannot.
#
# AND IT CHECKS SOMETHING STRONGER THAN ARITHMETIC. Every named external result below must still be
# named by at least one live test. A source that quietly stops being exercised is worse than a wrong
# count, because the claim it supports stays in the document either way. This list was built by
# probing candidates against real test names and `ags` FAILED THAT PROBE — the substring matched
# `flags` and `the_fundamental_matrix_tau_agrees_with...`, so a list assembled from memory would have
# certified the Amit–Gutfreund–Sompolinsky capacity as verified when nothing named it. The AGS result
# is here, under `replica`, which is what the test is actually called.
#
#   scripts/check-counts.sh             check the real documents
#   scripts/check-counts.sh --selftest  prove the check can say no
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=/Library/Developer/CommandLineTools/usr/bin:/opt/homebrew/bin:$PATH

MEMBERS=(ferrotherm ferrotherm-silicon ferrotherm-gpu ferrotherm-meter ferrotherm-serve ferrotherm-cloud)

# Named external results. Each entry is a substring that must appear in at least one test name, and
# each was verified to match the intended test rather than an unrelated word.
ORACLES=(onsager transfer_matrix gardner bethe curie_weiss replica krauth_mezard busclique
         elimination planar pfaffian katsura pfeuty nishimori wolff swendsen gauss_hermite
         landauer jarzynski)

# Names that mark a test as comparing against something outside the code's own behaviour.
VERIFY_VOCAB='exact|closed_form|onsager|enumerat|oracle|quadrature|transfer_matrix|bethe|gardner|analytic|brute_force|replica|krauth_mezard|busclique|pfaffian|katsura|pfeuty|nishimori|curie_weiss|closed form|known_value|matches_its|against_the'

names_of() { cargo test --release -p "$1" -- --list 2>/dev/null | grep ': test$' | sed 's/: test$//'; }

echo "deriving..."
LIB_NAMES=$(cargo test --release --lib -- --list 2>/dev/null | grep ': test$' | sed 's/: test$//')
LIB=$(printf '%s\n' "$LIB_NAMES" | grep -c .)

WS=0
for m in "${MEMBERS[@]}"; do
  n=$(names_of "$m" | grep -c .)
  if [ "$n" -eq 0 ]; then
    echo "member '$m' listed no tests at all -- it probably does not compile." >&2
    exit 2
  fi
  WS=$((WS + n))
done

VERIFY=$(printf '%s\n' "$LIB_NAMES" | grep -ciE "$VERIFY_VOCAB")

# EVERY NAMED SOURCE MUST STILL BE EXERCISED.
# NOT `grep -q` HERE. `grep -q` exits on its first match, the `printf` feeding it dies of SIGPIPE,
# and `set -o pipefail` reports that as a failed pipeline -- so a pattern that matches EARLY reports
# "not found" while one that matches late reports success. It is data-dependent and it silently
# inverted seven of these nineteen on the first run. Count instead; a count reads the whole stream.
missing=0
for o in "${ORACLES[@]}"; do
  hits=$(printf '%s\n' "$LIB_NAMES" | grep -ci -- "$o")
  if [ "$hits" -eq 0 ]; then
    echo "external source '$o' is named by NO live test -- either it stopped being checked, or the" >&2
    echo "  substring is wrong. Both are worse than a stale count." >&2
    missing=$((missing + 1))
  fi
done
[ "$missing" -gt 0 ] && { echo "$missing source(s) unexercised." >&2; exit 2; }
SOURCES=${#ORACLES[@]}

echo "  lib tests              $LIB"
echo "  workspace tests        $WS   (unit + integration, ${#MEMBERS[@]} crates)"
echo "  verification-bearing   $VERIFY"
echo "  external sources       $SOURCES   (each confirmed named by a live test)"

# --- the documents must agree ---------------------------------------------------------------
DOCS=(README.md LANDSCAPE.md)
bad=0
say() { echo "  $1" >&2; bad=$((bad + 1)); }

for d in "${DOCS[@]}"; do
  [ -f "$d" ] || { say "$d is missing"; continue; }
  grep -q "$WS tests" "$d" || say "$d does not quote the workspace test count ($WS tests)"
  # Stale figures must not survive anywhere, including in prose that was never updated.
  for stale in 699 911; do
    grep -q "$stale tests" "$d" && say "$d still quotes the stale count '$stale tests'"
  done
done
grep -q "$VERIFY " LANDSCAPE.md || say "LANDSCAPE.md does not quote the verification count ($VERIFY)"
grep -q "$SOURCES external sources" README.md || say "README.md does not quote '$SOURCES external sources'"
grep -q "$SOURCES external sources" LANDSCAPE.md || say "LANDSCAPE.md does not quote '$SOURCES external sources'"

if [ "$bad" -gt 0 ]; then
  echo >&2
  echo "$bad disagreement(s) between the documents and the suite." >&2
  echo "The suite is right. Update the documents to: $WS tests, $VERIFY verification-bearing," >&2
  echo "$SOURCES external sources." >&2
  exit 1
fi
echo "documents agree with the suite."

# --- prove it can say no ----------------------------------------------------------------------
if [ "${1:-}" = "--selftest" ]; then
  echo
  echo "selftest: a document with a wrong count must fail."
  tmp=$(mktemp -d)
  cp README.md "$tmp/README.md"
  sed -i.bak "s/$WS tests/123456 tests/" README.md
  if ./scripts/check-counts.sh >/dev/null 2>&1; then
    cp "$tmp/README.md" README.md
    echo "  THE CHECK PASSED ON A WRONG COUNT. It is not wired." >&2
    exit 2
  fi
  cp "$tmp/README.md" README.md
  rm -f README.md.bak
  echo "  it failed, as it must. The check is wired."
fi
