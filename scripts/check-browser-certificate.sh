#!/usr/bin/env bash
#
# Does the BROWSER sample the same distribution, or only find the same optimum?
#
# `check-answers.sh` already solves one model through the wasm build in a real browser engine and
# requires the same answer. That is a claim about ONE STATE. The roadmap's acceptance criterion for
# the deployment ladder is written about all of them -- "same distribution within the certificate's
# floor" -- and a sampler can agree about the optimum while disagreeing about everything else: the
# optimum is where the distribution is highest, not what shape it is.
#
# So this certifies the same graph on both sides and compares the certificates. `ising::ring(10, 1,
# 0.3)` at beta 0.5, seed 11, 3000 draws thinned by 8 -- ten spins, so the exact Boltzmann
# distribution is enumerable and `tv` is a real distance rather than an estimate.
#
# BOTH SIDES ENTER THROUGH THE C ABI. `examples/browser_parity.rs` calls `ft_builder_*` and
# `ft_certify` exactly as the page does, because a comparison whose two halves come in by different
# doors measures the doors as much as the library.
#
# ---- the tolerance, and where it came from -------------------------------------------------------
#
# MEASURED, not chosen. `exp` and `ln` come from the platform's libm and wasm32's need not agree
# with aarch64's in the last ulp; one ulp changes an acceptance, which changes a chain. The expected
# answer was therefore "two distributions that agree inside the noise floor". What was observed is
# much stronger: beta_eff, beta_lo, beta_hi, tau and ess are BIT-IDENTICAL, and `tv` differs by
# 2 ulps -- a relative 3.5e-16. The same chain, not merely the same distribution.
#
# TOL is set at 1e-9: seven orders of magnitude above what was measured, so a libm that shifts a few
# more ulps does not turn this red, and seven orders below a genuine divergence, which moves these
# quantities in the second decimal place. A threshold picked without a measurement behind it is a
# guess, and this project has shipped one of those before.
#
#   scripts/check-browser-certificate.sh              compare the two certificates
#   scripts/check-browser-certificate.sh --selftest   prove the comparison can fail
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

TOL=1e-9

if ! node scripts/cert-wasm.mjs --probe >/dev/null 2>&1; then
  if [ "${FERROTHERM_REQUIRE_ALL:-}" = "1" ]; then
    echo "playwright is not installed and FERROTHERM_REQUIRE_ALL=1, so this is a failure" >&2
    exit 1
  fi
  echo "skipped: no playwright, so the browser half cannot run"
  echo "  (CI sets FERROTHERM_REQUIRE_ALL=1, where this skip is a failure)"
  exit 0
fi

native="$(cargo run --quiet --example browser_parity 2>/dev/null | tail -1)"
if [ -z "$native" ]; then
  echo "the native half produced nothing; it did not certify" >&2
  exit 1
fi
wasm="$(node scripts/cert-wasm.mjs 2>/dev/null | tail -1)"
if [ -z "$wasm" ]; then
  echo "the browser half produced nothing; it did not certify" >&2
  exit 1
fi

if [ "${1:-}" = "--selftest" ]; then
  # Damage ONE field of the browser's certificate by a relative 1e-6 -- far below anything a reader
  # would notice, and far above the 3.5e-16 the two really differ by. A gate that prints "the
  # browser samples what the native build samples" is printing exactly what it would print if the
  # comparison had stopped comparing, so it has to be shown saying no.
  damaged="$(python3 -c "
import json,sys
c = json.loads(sys.argv[1]); c['beta_eff'] *= 1.000001; print(json.dumps(c))" "$wasm")"
  if python3 scripts/_certificates_agree.py "$native" "$damaged" "$TOL" >/dev/null 2>&1; then
    echo "SELFTEST FAILED: a certificate off by 1e-6 compared EQUAL, so this gate proves nothing" >&2
    exit 1
  fi
  echo "selftest: a certificate nudged by 1e-6 was caught, so the comparison can fail"
  exit 0
fi

python3 scripts/_certificates_agree.py "$native" "$wasm" "$TOL"
