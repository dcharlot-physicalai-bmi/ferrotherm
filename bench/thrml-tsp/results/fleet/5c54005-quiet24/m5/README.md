# M5 Max, harness 5c54005

Host `Davids-MacBook-Pro-M5-Max-48GB-4.local`, macOS 27.2, Apple M5 Max, 18
cores. macmon 0.11.0, `backend=macmon`, `machine=Apple M5 Max`,
`scope=soc-telemetry`. That scope is SoC telemetry, not wall power. THRML
0.1.4, JAX 0.12.0.1. Harness `5c54005fd0ff8dc62adedda478faa52388911902`,
`git_dirty` false.

Label for attempt 2: **preflight pass, energy refused (load1 2.4 in baseline)**.

The files are published as written. `attempt-002/report.json` stamps
`evidence_grade` `quiet` and `energy.outcome` `unavailable`. That stamp is the
grading defect: the preflight held, and the energy phase did not. The note on
the energy object is the meter's refusal that the 1-minute load average was
2.4 during the idle baseline. `idle_watts` is null. `summaries` is empty.
`trials` is empty. No joules are invented from the macmon trace.

`exit.txt` is `0 grade=quiet attempt=2`. That line is the same defect. The
grade of this attempt is the label above, not quiet.

Attempt 1 is `load_gated`. `quiet-gate.json` has `held` false, `max_load1`
2.3, threshold 2, 60.652 s. Attempt 2 preflight held: `max_load1` 1.88,
`load1` at the start of the receipt 1.37, threshold 2, 60.660 s,
`busy_readable` false. `phase_order` is energy then accuracy. Fifteen accuracy
workloads completed. Energy took no trials.

Timing from `events.jsonl`, UTC. Driver pid 34577 started
`2026-11-09T19:40:32.048644+00:00` (2:40:32 PM EST). Attempt 2 ran
`2026-11-09T19:50:32.059998+00:00` to `2026-11-09T19:54:24.762252+00:00`
(2:50:32 PM to 2:54:24 PM EST). `driver_end` is
`2026-11-09T19:54:24.765226+00:00`.

Not in this tree, and not reconstructed: `attempt-002/thrml_raw.txt` and the
`ising_open_32x32` and `ising_open_64x64` `states.txt` traces. They remain on
the Mac at `~/bench-scratch/evidence-5c54005-quiet24-m5/`.
