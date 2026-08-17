# Malformed input to ferrotherm's parsers aborts the calling process

```toml
[advisory]
id = "RUSTSEC-0000-0000"
package = "ferrotherm"
date = "2026-08-17"
url = "https://github.com/dcharlot-physicalai-bmi/ferrotherm/releases/tag/v0.13.0"
references = [
    "https://github.com/dcharlot-physicalai-bmi/ferrotherm/releases/tag/v0.14.0",
    "https://github.com/dcharlot-physicalai-bmi/ferrotherm/releases/tag/v0.15.0",
]
categories = ["denial-of-service"]
keywords = ["parser", "panic", "abort", "ffi", "protobuf"]

[versions]
patched = [">= 0.14.0"]
```

## Summary

`ferrotherm`'s parsers panic on small malformed inputs. Because the crate's parsing entry points are
also exposed through a C ABI (`ft_ommx_read`, `ft_ising2d_new`, `ft_model_integer`,
`ft_planted_wishart`), and **a Rust panic across an `extern "C"` boundary is non-unwinding**, the
panic is an `abort`: it terminates the process that linked the library rather than returning an error
the caller can handle. Callers from C, Python `ctypes`, Julia `ccall`, Zig and WebAssembly are all
affected.

Two further inputs cause unbounded allocation rather than a panic.

## Details

Reachable from untrusted input, with the smallest reproducer measured for each:

| input | bytes | effect |
|---|---|---|
| OMMX instance whose length prefix overflows `usize` | 11 | the wrapped length passes the bounds check; slice panic |
| OMMX instance with a diagonal quadratic term (`row == col`) | 23 | reaches the graph builder as a self-edge; panic. **This is well-formed OMMX** that other tools emit routinely |
| `.ftp` program with a colour index of `u64::MAX` | 45 | `c + 1` overflows |
| `.ftp` program declaring `spins 18446744073709551615` with a large colour index | ~50 | ~96 GB allocation; the `c < spins` bound is vacuous when `spins` is unbounded |
| LP file declaring an integer over most of `i64` | 6 lines | one objective term emitted per value in the domain; ~1 GB and rising |
| `ft_ising2d_new(1, ..)` | one call | the periodic boundary wraps onto the site itself, building a self-edge |
| `ft_planted_wishart(3, f64::INFINITY, ..)` | one call | the guard refused `NaN` and admitted `+inf`; capacity overflow |
| `ft_model_integer` with a range spanning most of `i64` | two calls | reported success, then `ft_model_compile` aborted |

## Impact

Denial of service. A process that parses an untrusted `.ftp`, LP or OMMX file — or that exposes any
of these entry points to input it does not control — can be terminated by a few dozen bytes. No
memory-safety violation or code execution is involved.

## Patched

`0.14.0` fixes all of the above. `0.15.0` additionally rewrites the protobuf decoder so that a
truncated message is an error rather than being silently read as a shorter valid one; that defect
causes misinterpretation rather than a crash, so it is not part of this advisory.
