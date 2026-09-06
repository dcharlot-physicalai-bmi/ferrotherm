#!/usr/bin/env bash
#
# The lints this crate has NOT finished, counted -- and a ratchet so the count can only fall.
#
# `cargo clippy -W clippy::pedantic -W clippy::nursery` reports thousands of findings on any real
# numerics codebase, and there are exactly two dishonest ways to deal with that. One is to fix all of
# them, which spends a week rewriting correct code and, for several lints here, actively breaks it --
# `suboptimal_flops` fuses multiply-add and changes results this crate verifies at 1e-9. The other is
# to allow them silently, which turns a real gap into an invisible one.
#
# So: `Cargo.toml` DENIES the set that was worth fixing (fixed, and CI runs at -D warnings, so that
# count is zero by construction). The families below are the ones that are real, large, and cannot be
# honestly closed in a sitting -- writing 509 doc comments in an afternoon produces 509 sentences
# that say nothing. They are counted here instead, with the count committed, and this gate fails if
# any of them RISES. New code carries its docs; the backlog comes down a release at a time.
#
#   scripts/check-lints.sh              counts must not exceed the recorded baselines
#   scripts/check-lints.sh --update     record the current counts (only ever to LOWER them)
#   scripts/check-lints.sh --selftest   prove the ratchet can fail
#
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
baseline="$here/scripts/lint-baseline.txt"

count() { # $1 = clippy flag, $2 = the phrase its finding prints
  cargo clippy --release -p ferrotherm --all-targets -- "$1" 2>&1 | grep -c "$2"
}
measure() {
  echo "missing_errors_doc $(count -Wclippy::missing_errors_doc 'missing .# Errors')"
  echo "missing_panics_doc $(count -Wclippy::missing_panics_doc 'missing .# Panics')"
}

if [[ "${1:-}" == "--update" ]]; then
  measure > "$baseline"
  echo "recorded:"; sed 's/^/  /' "$baseline"
  echo "Commit this. A number that went UP is a regression, not a new baseline."
  exit 0
fi

[[ -f "$baseline" ]] || { echo "no baseline at $baseline; run --update once and commit it" >&2; exit 2; }

if [[ "${1:-}" == "--selftest" ]]; then
  # Damage the BASELINE, not the code: claim a count one lower than reality and require a failure.
  # A ratchet that cannot fire is a file nobody reads.
  tmp=$(mktemp) || exit 2
  trap 'rm -f "$tmp"' EXIT
  first=$(head -1 "$baseline"); name=${first%% *}; n=${first##* }
  printf '%s %d\n' "$name" "$((n - 1))" > "$tmp"
  tail -n +2 "$baseline" >> "$tmp"
  # Not `measure | head -1`: `head` closes the pipe and the remaining counts die on SIGPIPE, which
  # prints a shell error into the middle of a passing selftest.
  all=$(measure); now=$(printf '%s\n' "$all" | sed -n 1p); nowname=${now%% *}; nownum=${now##* }
  want=$(grep "^$name " "$tmp" | cut -d' ' -f2)
  if [[ "$nownum" -le "$want" ]]; then
    echo "SELFTEST FAILED: $name measured $nownum against a deliberately-lowered baseline of $want," >&2
    echo "                 so a real regression of one would slip through this comparison." >&2
    exit 1
  fi
  echo "selftest ok: a baseline lowered by one is detected ($nowname $nownum > $want), so the ratchet fires."
  exit 0
fi

fail=0
while read -r name want; do
  [[ -n "$name" ]] || continue
  case "$name" in
    missing_errors_doc)  got=$(count -Wclippy::missing_errors_doc 'missing .# Errors') ;;
    missing_panics_doc)  got=$(count -Wclippy::missing_panics_doc 'missing .# Panics') ;;
    *) echo "unknown lint family in the baseline: $name" >&2; exit 2 ;;
  esac
  if [[ "$got" -gt "$want" ]]; then
    printf '  %-20s %4d  ** was %d -- this went UP **\n' "$name" "$got" "$want"
    fail=1
  elif [[ "$got" -lt "$want" ]]; then
    printf '  %-20s %4d  (was %d -- run --update to bank it)\n' "$name" "$got" "$want"
  else
    printf '  %-20s %4d  unchanged\n' "$name" "$got"
  fi
done < "$baseline"

if [[ $fail -ne 0 ]]; then
  echo >&2
  echo "a tracked lint family grew. New code carries its own docs; the backlog only comes down." >&2
  exit 1
fi
echo "  every tracked family is at or below its baseline."
