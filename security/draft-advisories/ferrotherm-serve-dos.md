# Unauthenticated remote denial of service in ferrotherm-serve

```toml
[advisory]
id = "RUSTSEC-0000-0000"
package = "ferrotherm-serve"
date = "2026-08-17"
url = "https://github.com/dcharlot-physicalai-bmi/ferrotherm/releases/tag/v0.13.0"
categories = ["denial-of-service"]
keywords = ["http", "json", "stack-overflow", "dos", "unauthenticated"]
cvss = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H"

[versions]
patched = [">= 0.7.2"]
```

## Summary

`ferrotherm-serve` exposes an HTTP API that reads request bodies from the network without
authentication. Three unbounded paths let a single small request consume the server:

1. **Stack exhaustion.** The JSON parser was recursive descent with no depth limit. A **40 KB** body
   of nested arrays overflowed the thread stack and **aborted the whole process** — not the request.
   The next connection received `ECONNREFUSED`.
2. **Wrapping arithmetic in the `/v1/anneal` budget.** `(stages * per) as u64` multiplies in `usize`
   *before* the cast, so `"stages": 9223372036854775808` produced a small value, passed the
   node-update ceiling, and aborted in `raw_vec` with a capacity overflow. The client received an
   empty reply rather than a `400`.
3. **No effective ceiling on `/v1/solve`.** Its update bound sat inside the schedule branch, so a
   request naming no schedule had none — and no ceiling measured the dimension that actually grows.
   `{"variables":[{"name":"x","values":1000}],"tries":1}` is **46 bytes** and compiles to only 1000
   spins, far under the node limit, while a one-hot over *k* values carries *k(k−1)/2* couplings:
   499,500 of them, 6.7 s of CPU and a 17 MB response.

## Impact

Any unauthenticated client that can reach the listener can terminate the server (1, 2) or exhaust CPU
and memory (3). No authentication is required and no memory-safety violation is involved.

## Patched

`0.7.2` adds a JSON nesting limit of 64, saturates the anneal budget in `u64` and clamps the ladder,
and bounds `/v1/solve` by compiled coupling count with a ceiling derived from measurement rather than
guessed.
