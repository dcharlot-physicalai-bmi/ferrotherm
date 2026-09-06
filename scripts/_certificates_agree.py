#!/usr/bin/env python3
"""Compare two certificates of the same run taken on two builds.

Used by ``check-browser-certificate.sh``; separate so its selftest can drive the comparison with a
deliberately damaged input without re-running either build.

Two verdicts, and they answer different questions:

* the roadmap's acceptance criterion, which is about DISTRIBUTIONS -- the two `tv` figures must be
  closer to each other than the certificate's own sampling-noise floor, since a difference below
  that floor is not a difference this instrument can see;
* a much tighter regression bound on every field, because what these two builds actually produce is
  the same chain, and silently losing that would be worth knowing about.
"""

import json
import math
import sys

FIELDS = ("beta_eff", "beta_lo", "beta_hi", "tau", "ess", "tv", "floor")


def rel(a, b):
    """Relative difference, falling back to absolute where the scale is degenerate."""
    scale = max(abs(a), abs(b))
    return abs(a - b) / scale if scale > 1e-300 else abs(a - b)


def main():
    native, wasm, tol = json.loads(sys.argv[1]), json.loads(sys.argv[2]), float(sys.argv[3])

    # NON-FINITE FIRST, because every comparison below silently passes on NaN.
    #
    # `ft_cert_tv` and `ft_cert_floor` return `tv_exact.unwrap_or(NAN)` and
    # `noise_floor.unwrap_or(NAN)`: a graph too wide to enumerate exactly has no `tv` at all. If one
    # side ever came back that way -- which is precisely the kind of divergence between a 64-bit
    # host and a 32-bit wasm target this gate exists to catch -- then `spread >= floor` is
    # `nan >= nan`, which is False, and `worst > tol` is `nan > tol`, also False. Both verdicts pass
    # and the gate prints "the same chain" about a certificate carrying no measurement whatsoever.
    for name, c in (("native", native), ("browser", wasm)):
        bad = [f for f in FIELDS if not isinstance(c.get(f), (int, float))
               or not math.isfinite(float(c[f]))]
        if bad:
            print(f"the {name} certificate has no measurement for {', '.join(bad)} -- there is "
                  f"nothing to compare, and a comparison against a non-number silently passes",
                  file=sys.stderr)
            return 1

    for name, c in (("native", native), ("browser", wasm)):
        if not c.get("passed"):
            print(f"the {name} certificate did not pass, so there is nothing to compare: "
                  f"{c.get('findings')} finding(s)", file=sys.stderr)
            return 1

    # The criterion the roadmap is written in. Stated first because it is the claim; the bound
    # below is the regression guard.
    floor = min(native["floor"], wasm["floor"])
    spread = abs(native["tv"] - wasm["tv"])
    if spread >= floor:
        print(f"the two builds sample distributions {spread:.4f} apart against a {floor:.4f} "
              f"sampling-noise floor -- further apart than this instrument can call equal",
              file=sys.stderr)
        return 1

    worst, worst_at = 0.0, None
    for f in FIELDS:
        d = rel(native[f], wasm[f])
        if d > worst:
            worst, worst_at = d, f
    if worst > tol:
        print(f"the browser and native certificates differ by {worst:.3e} at {worst_at} "
              f"(native {native[worst_at]!r}, browser {wasm[worst_at]!r}), over a {tol:.0e} bound",
              file=sys.stderr)
        return 1

    print(f"one graph certified on two builds: tv {native['tv']:.4f} native, {wasm['tv']:.4f} in "
          f"the browser, {spread:.2e} apart against a {floor:.4f} noise floor")
    print(f"  and they agree far more tightly than that -- worst field {worst:.2e} at "
          f"{worst_at or 'none'}, so this is the same chain, not merely the same distribution")
    print(f"  beta_eff {native['beta_eff']:.6f} (asked 0.500000), ess {native['ess']:.1f}, "
          f"tau {native['tau']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
