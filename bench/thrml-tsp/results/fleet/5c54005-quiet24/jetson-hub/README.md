# jetson-hub, attempt 2

Harness `5c54005fd0ff8dc62adedda478faa52388911902`, `git_dirty` false. The
driver event names the host `jetson-hub`. The receipt's CPU is a 13th Gen
Intel Core i9-13900H. The driver pinned cpus `8-11`. Meter is RAPL:
`package-0` and the `core` subdomain, recorded separately and not added. The
receipt also sees an NVIDIA GeForce RTX 4050 via NVML and does not add it to
the package counter. No joules were taken: the preflight refused before a
trial. This directory is not pooled with the M5 `soc-telemetry` refusal or
with `framework-128gb-0000`.

Evidence grade **load_gated**. `attempt-002/quiet-gate.json`: `held` false,
`max_load1` 5.24, threshold 2, 60.01 s, `busy_readable` true. `load1` at the
start of the receipt is 0.18, and the 60 s window still reached 5.24.
`attempt-002/report.json` has `energy` null and no workloads. The gate was
not loosened.

Driver `python3 run-5c54005-quiet24-jetsonhub.py`, pid 2534843, `driver_start`
`2026-11-09T21:06:23.965925+00:00` (4:06:23 PM EST). Attempt 2 is
`2026-11-09T21:16:23.966391+00:00` to `2026-11-09T21:17:31.687859+00:00`, exit
1. There is no `exit.txt`.
