#!/usr/bin/env bash
#
# The picture and the API are the same model. Are they the same ANSWER?
#
# check-editor-parity.sh proves the editor can SAY every constraint the model layer has. That is a
# check on vocabulary, and it passes an editor whose `fromModel` drops a k, a soft price or an
# objective term -- such an editor still draws every node type, still runs, and answers a different
# question. The gap between "can say it" and "means it" is exactly where check-parity.sh and
# check-semantics.sh sit apart for the language bindings, and the editor deserves the same pair.
#
# So: one JSON model at a time, through the HTTP API and through the editor's fromModel, comparing
# the COMPILED size and the feasibility. Not the values -- both surfaces anneal, and two runs of a
# stochastic sampler are entitled to differ. Spins and ancillas are not: they are what the model
# compiled to, and a lost constraint shows there.
#
#   scripts/check-editor-model.sh
#
# --selftest first: this gate reads its numbers out of a report with a regex, and a regex that stops
# matching reports -1 on both sides and calls it agreement.

set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

node scripts/editor-vs-api.mjs --probe 2>/dev/null || {
  echo "  skipped: playwright is not installed (web-tests/node_modules)"; exit 0; }

cargo build --quiet -p ferrotherm-serve --bin ferrotherm-serve || {
  echo "  the server did not build" >&2; exit 2; }

PORT="${FT_PORT:-8481}"
./target/debug/ferrotherm-serve "127.0.0.1:$PORT" >/dev/null 2>&1 &
srv=$!
trap 'kill $srv 2>/dev/null' EXIT

# Wait for it to answer rather than sleeping a guess: a fixed sleep is either slow or flaky, and on
# a loaded runner it is both.
for _ in $(seq 1 50); do
  curl -fsS -X POST "http://127.0.0.1:$PORT/v1/capabilities" -d '{}' >/dev/null 2>&1 && break
  sleep 0.2
done

export FT_SERVE="127.0.0.1:$PORT"
echo "selftest"
node scripts/editor-vs-api.mjs --selftest >/dev/null 2>&1 || {
  echo "  SELFTEST FAILED: a damaged editor agreed with the API on every model"; exit 1; }
echo "  ok      an editor that drops a k is caught"
echo "models"
node scripts/editor-vs-api.mjs
