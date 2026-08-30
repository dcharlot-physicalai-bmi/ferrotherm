#!/usr/bin/env bash
#
# The browser pages must actually run. Both of them.
#
# THIS EXISTS BECAUSE `docs/ide.html` HAD NO GATE. `check-editor-parity.sh` and
# `check-editor-model.sh` both drive `docs/graph.html`; nothing looked at the workbench. An editing
# slice deleted two of its functions -- `showFit` and `fitMessage`, both called from `apply()` --
# and the page threw "showFit is not defined" on every run, in two published releases and on the
# live site, with twelve gates green the whole time.
#
# WHY THIS IS A WRAPPER AND NOT A STATIC CHECK. The obvious gate is to resolve the linkage by hand:
# collect every function a page defines, collect every one it calls, and diff. That was written
# first and thrown away. These pages carry embedded CSS and long prose comments, so `rgba(`,
# `translateX(`, `@media (` and ordinary English inside a comment all read as calls to a regex, and
# the first run produced sixty-odd false names across three files. A gate needing thirty exemptions
# is a gate nobody reads, and one whose pattern is loose enough to avoid them would have let
# `showFit` through -- which is the only defect it was written for.
#
# The behavioural suites already resolve it correctly, by running the page. What was missing was
# never the check; it was that `web-tests/` ran ONLY in CI, and a hand-run of `node editor.test.mjs`
# covered one of the two files. This makes both reachable as a gate, so "did I run the pages" stops
# depending on remembering which suite is which.
#
# A SYNTAX CHECK WOULD NOT HAVE HELPED, and that is worth stating where the next person will read
# it: `new Function(src)` was run over these scripts after the edit that broke them and passed. A
# call to a function that does not exist is valid JavaScript right up until it runs.
#
#   scripts/check-pages.sh              run every browser suite
#   scripts/check-pages.sh --selftest   prove the comparison can fail
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

# Locally a skip is right -- not every machine has playwright installed. In CI it is the failure
# mode this gate exists to catch, so `FERROTHERM_REQUIRE_ALL=1` turns the skip into a failure. Same
# contract as `check-answers.sh`, which learned it the same way.
have_playwright() {
  node -e 'require.resolve("playwright", { paths: ["'"$here"'/web-tests"] })' >/dev/null 2>&1
}

if ! have_playwright; then
  if [ "${FERROTHERM_REQUIRE_ALL:-}" = "1" ]; then
    echo "playwright is not installed and FERROTHERM_REQUIRE_ALL=1, so this is a failure:" >&2
    echo "  cd web-tests && npm install && npx playwright install --with-deps chromium" >&2
    exit 1
  fi
  echo "skipped: no playwright in web-tests/node_modules"
  echo "  cd web-tests && npm install && npx playwright install --with-deps chromium"
  echo "  (CI sets FERROTHERM_REQUIRE_ALL=1, where this skip is a failure)"
  exit 0
fi

SUITES="editor.test.mjs workbench.test.mjs"

if [ "${1:-}" = "--selftest" ]; then
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  cp docs/ide.html "$tmp/ide.html.orig"

  # THE DAMAGE IS THE REAL REGRESSION, not a strawman: delete a function that is still called, which
  # is exactly what the editing slice did. Truncating the file or breaking its syntax would be
  # caught by anything; a call into nothing is caught by running the page and by nothing else.
  python3 - docs/ide.html <<'PY'
import sys
p = sys.argv[1]
t = open(p).read()
i = t.index('function showFit()')
depth, k = 0, t.index('{', i)
while True:
    if t[k] == '{':
        depth += 1
    elif t[k] == '}':
        depth -= 1
        if depth == 0:
            break
    k += 1
open(p, 'w').write(t[:i] + t[k + 1:])
PY
  restore() { cp "$tmp/ide.html.orig" docs/ide.html; }
  trap 'restore; rm -rf "$tmp"' EXIT

  if grep -q 'function showFit()' docs/ide.html; then
    echo "SELFTEST FAILED: the damage did not apply -- showFit is still defined" >&2
    exit 1
  fi
  if ! grep -q 'showFit()' docs/ide.html; then
    echo "SELFTEST FAILED: the damage removed the CALL as well, so there is nothing to detect" >&2
    exit 1
  fi

  if (cd web-tests && node workbench.test.mjs >/dev/null 2>&1); then
    echo "SELFTEST FAILED: the workbench suite passed a page whose showFit was deleted while" >&2
    echo "still being called. That is the defect that shipped in two releases." >&2
    exit 1
  fi
  echo "selftest: a deleted-but-still-called function was caught, so this gate can fail"
  exit 0
fi

for suite in $SUITES; do
  printf '  %-22s ' "$suite"
  if (cd web-tests && node "$suite" >"$here/.pagelog" 2>&1); then
    printf 'pass (%s checks)\n' "$(grep -c '^PASS' "$here/.pagelog" || echo '?')"
  else
    echo "FAILED"
    grep -E '^FAIL' "$here/.pagelog" | head -20 >&2
    rm -f "$here/.pagelog"
    exit 1
  fi
done
rm -f "$here/.pagelog"
echo "both browser pages run: every function they call exists and every claim they make holds"
