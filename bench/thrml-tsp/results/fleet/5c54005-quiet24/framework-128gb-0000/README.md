# framework-128gb-0000, attempt 10

Harness `5c54005fd0ff8dc62adedda478faa52388911902`, `git_dirty` false. Host CPU
in the receipt: AMD Ryzen AI MAX+ 395 w/ Radeon 8060S. The driver pinned
`4-7,20-23` and asked for `--device cpu`. Meter is RAPL: `package-0` and the
`core` subdomain, recorded separately and not added. There is no macmon row
and no NVML device. This directory is not pooled with the M5 `soc-telemetry`
refusal or with `jetson-hub`.

Evidence grade **load_gated**. `attempt-010/quiet-gate.json`: `held` false,
`max_load1` 22.47, threshold 2, 60.014 s, `busy_readable` true. `load1` at the
start of the receipt is 20.32. `attempt-010/report.json` has `energy` null and
no workloads. No joules were taken. The gate was not loosened.

Driver `python3 run-5c54005-quiet24.py`, pid 2903913, `driver_start`
`2026-11-09T19:40:16.953529+00:00` (2:40:16 PM EST). Attempt 10 itself is
`2026-11-09T21:10:16.953758+00:00` to `2026-11-09T21:11:19.600867+00:00`, exit
1. `events.jsonl` also contains earlier attempts. The refusal published here
is attempt 10. There is no `exit.txt`.
