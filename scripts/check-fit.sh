#!/usr/bin/env bash
# Fitting a model to data must give the SAME answer in the browser as on the machine.
#
# The other surfaces each test their own binding. This one asks the harder question -- does the wasm
# the page actually loads produce the same fit as the native library -- because that is where a
# fitting stack quietly stops being one. A browser that trains to a different optimum than the CLI
# is not a portable paradigm; it is two products with one name.
#
# The comparison is EXACT. Contrastive divergence here is scalar IEEE-754 arithmetic over a seeded
# PCG stream, with no threading, no reductions and no kernel selection, so there is no licence for
# the two to differ at all. A tolerance would hide the class of bug this exists to catch: a
# marshalling error that shifts one row, or a stale committed wasm serving an older sampler.
#
#   scripts/check-fit.sh              compare the browser binary against the native library
#   scripts/check-fit.sh --selftest   prove the comparison can fail
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

SEED=3
SELFTEST=0
if [ "${1:-}" = "--selftest" ]; then SELFTEST=1; fi

lib="$here/target/release/libferrotherm.dylib"
[ -f "$lib" ] || lib="$here/target/release/libferrotherm.so"
if [ ! -f "$lib" ]; then
  echo "no native library at target/release; run: cargo build --release" >&2
  exit 2
fi
if [ ! -f "$here/docs/ferrotherm.wasm" ]; then
  echo "no docs/ferrotherm.wasm; run: cargo build --release --lib --target wasm32-unknown-unknown" >&2
  exit 2
fi

# The browser arm. A DIFFERENT seed under --selftest, which is the whole point: the check must be
# able to fail, and the cheapest true difference is a fit that ran from another stream.
wasm_seed=$SEED
if [ "$SELFTEST" = 1 ]; then wasm_seed=4; fi
browser="$(node "$here/scripts/fit-wasm.mjs" --seed "$wasm_seed")"

# The native arm, through the Python binding, which loads the same dylib every other native caller
# does. Going through a binding rather than a Rust test is deliberate: it puts a real marshalling
# layer on both sides of the comparison instead of only on the browser's.
native="$(PYTHONPATH="$here/python" python3 - "$SEED" <<'PY'
import json, math, sys
import ferrotherm as ft
seed = int(sys.argv[1])
rows = ft.bars_and_stripes(3)
visible = len(rows[0])
floor, ceiling = -visible * math.log(2), -math.log(len(rows))

def fit(sim, label):
    before = sim.log_likelihood(rows)
    sim.fit(rows, epochs=400, k=10, seed=seed)
    after = sim.log_likelihood(rows)
    return {"label": label, "spins": len(sim), "before": before, "after": after,
            "learned": (after - floor) / (ceiling - floor) * 100}

out = {"rows": len(rows), "visible": visible,
       "wide": fit(ft.rbm(visible, 12, seed=1), "wide"),
       "deep": fit(ft.dbm(visible, [6, 6], seed=1), "deep")}
print(json.dumps(out))
PY
)"

status=0
if ! python3 - "$browser" "$native" <<'PY'
import json, math, sys
b, n = json.loads(sys.argv[1]), json.loads(sys.argv[2])
bad = []

if b["rows"] != n["rows"] or b["visible"] != n["visible"]:
    bad.append(f'the two arms disagree about the dataset: {b["rows"]}x{b["visible"]} vs '
               f'{n["rows"]}x{n["visible"]}')

# The untrained end of the scale is not measured, it is DERIVED: every weight is zero, so the model
# is uniform over 2^visible images and the likelihood can only be -visible*ln2. A surface that gets
# this wrong has a marshalling bug, not a training bug, and would otherwise look like a bad fit.
floor = -b["visible"] * math.log(2)
for arm in ("wide", "deep"):
    for who, d in (("browser", b), ("native", n)):
        if abs(d[arm]["before"] - floor) > 1e-12:
            bad.append(f'{who} {arm}: an untrained model must score exactly {floor:.15f}, '
                       f'and it scored {d[arm]["before"]:.15f}')

for arm in ("wide", "deep"):
    x, y = b[arm], n[arm]
    if x["spins"] != y["spins"]:
        bad.append(f'{arm}: {x["spins"]} spins in the browser and {y["spins"]} natively')
    if x["after"] != y["after"]:
        bad.append(f'{arm}: the browser fitted to {x["after"]!r} and the native library to '
                   f'{y["after"]!r} -- scalar IEEE-754 over a seeded stream has no licence to differ')
    # A regression guard on the fit itself, not a claim about depth: this preset is the one the
    # workbench ships, and a version of it that learns half as much is a broken default.
    if y["learned"] < 85.0:
        bad.append(f'{arm}: fitted to only {y["learned"]:.1f}% of the way to a perfect model')

if bad:
    for line in bad:
        print("  " + line)
    sys.exit(1)
print(f'  browser and native agree exactly: wide {b["wide"]["after"]!r} '
      f'({b["wide"]["learned"]:.1f}% learned), deep {b["deep"]["after"]!r} '
      f'({b["deep"]["learned"]:.1f}% learned)')
PY
then status=1; fi

if [ "$SELFTEST" = 1 ]; then
  if [ "$status" = 0 ]; then
    echo "SELFTEST FAILED: a fit from a different seed compared EQUAL, so this check proves nothing" >&2
    exit 1
  fi
  echo "selftest: a fit from another seed was caught, so the comparison can fail"
  exit 0
fi

if [ "$status" != 0 ]; then
  echo "the browser and the native library do not fit alike" >&2
  exit 1
fi
echo "fitting agrees across the browser and the native library, and both hit the exact untrained floor"
