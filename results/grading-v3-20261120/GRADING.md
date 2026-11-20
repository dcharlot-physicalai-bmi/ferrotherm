# ferrotherm energy grading, rules v3: read-only pass (2026-11-20)

- Rules: METHODOLOGY.md v3 at freeze commit dbf9653a6091543df420f9ab8a3fe5da9b00a758. Harness: 777f0417340788428cf0525003277f9829ca6e3a (perf/large-grid).
- Scope: energy campaign logs that ran until the 2026-11-15T21:30Z deadline on the M5 48GB, thor.local, jetson-hub and framework-128gb-0000.
- Method: read only. Logs were copied with rsync/scp/tar from the hosts. No process was signalled, no config was changed, no sudo or powermetrics was used, and thor2.local and the Forgejo CI jobs were not touched.
- Grading: per meter, never pooled. A meter is graded only on its own idle brackets with its own 3σ test. A meter with no bracket of its own is ungraded. Parity line 0.05, unchanged.
- Labels: c82c1e96 (pre-v3) attempts are labelled **regrade-v3** and kept separate from **native-v3** (777f0417) attempts.
- Baselines: THRML 0.1.4 = baseline A. THRML main 2d96f4b4 = baseline B.
- 8e9ad0a4 energy is not cited. jetson-hub attempt 037 (c82c1e96, Intel RAPL core-domain J/ESS vs THRML 0.1.4 on JAX CPU, per workload; tarball sha 1df27d9f, publish 22d3e720; in-window scoring defect counts against ferrotherm; 4x4 row under investigation) is referenced only and was not re-graded.

## Result: nothing passes

No meter on any host has a trial that passes v3, so every `*-passed.csv` is header-only.

| Host | Meter | Rules | Attempts examined | Reached energy phase | Passed | Refused |
|---|---|---|---|---|---|---|
| M5 Max 48GB | macmon SoC telemetry | regrade-v3 (c82c1e96) | 338 | 42 | 0 | 338 |
| thor.local | ina238 2-0044 VIN (module input, not wall) | regrade-v3 (c82c1e96) | 115 | 2 (no trials) | 0 | 115 |
| jetson-hub | RAPL package | native-v3 (777f0417) | 15 | 1 (+1 incomplete) | 0 | 15 |
| jetson-hub | RAPL core | native-v3 (777f0417) | 15 | 1 (+1 incomplete) | 0 | 15 |
| jetson-hub | NVML RTX 4050 | native-v3 (777f0417) | 15 | 1 (+1 incomplete) | 0 | 15 |
| framework-128gb-0000 | AMD RAPL package (model-based) | regrade-v3 (c82c1e96) | 39 | 1 | 0 | 39 |
| framework-128gb-0000 | AMD RAPL core (model-based) | regrade-v3 (c82c1e96) | 39 | 1 | 0 | 39 |

There are no passing J/ESS values, so no ratio against baseline A or B is reported for any meter.

## Refusals by host and meter (full rows in `receipt/<host>/<meter>-refusals.csv`)

### M5 48GB, macmon

Driver PID 3596 exited. exit.txt: `deadline attempts=338 no_quiet_lock_held_joules` (written 2026-11-15 16:26 ET; driver_end 21:26:54Z).

- 296 attempts: load_gated at the preflight quiet gate (load1 > 2).
- 42 attempts: quiet preflight, then energy load_gated.
  - 46 trials examined, 0 attributable.
  - Trial-window refusals: 30 for "a sample exceeded the maximum", 16 for "the mean exceeded the maximum".
  - Attempt-level reasons: bracket drift > 3σ (10), idle bracket load1 not ≤ 2 for 60 s (9), load average > 2 at the bracket (10), and the trial-window rule alone (13).
- All 42 also fail regrade-v3 structurally:
  - Rule 1: no window_audit, and the metered window is not sampling-only.
  - Rule 3: brackets are per A/B pair, with no bracket between paired trials.
- The early attempts did not record host_lock_held (30 attempts, None). It was true from the restart onward.

### thor.local, ina238 VIN

- The quiet24 c82c1e96 driver, PID 1341477 (restarted 2026-11-13T13:29Z), is no longer running. Its only attempt was 115, which was load_gated at 13:30Z. After that there was no driver_end event and no new exit.txt.
- exit.txt is stale from the previous run: `deadline attempts=114 no_quiet_lock_held_joules` (2026-11-10T21:29Z).
- thor rebooted on 2026-11-13 at 14:28 UTC (and several times later). The driver did not survive, so thor recorded no attempts from Nov 13 13:30Z to the deadline.
- 113 attempts were load_gated at preflight.
- Attempts 8 and 20 were refused because the lock was not held (waited 3600 s; energy load_gated, 0 trials).
- No attributable trials.

### jetson-hub, native v3 at 777f0417 (RAPL package, RAPL core and NVML graded separately)

- Driver 2288213 is not running, and there is **no exit.txt**.
  - The last event is the start of attempt 15 at 2026-11-13T16:12Z. The attempt-015 log was last written at 2026-11-14T15:21Z.
  - The host rebooted on 2026-11-16 at 15:51 ET. The driver never reached the deadline.
- Attempts 1–6 and 8–14: load_gated at preflight. Attempt 2 also waited 997 s on the lock held by the parity queue.
- Attempt 7: quiet preflight, lock held. The suite half was refused because its idle bracket was refused (load1 not ≤ 2 for 60 s), with 0 trials. The grid half produced one ferrotherm-gpu trial for grid_scale_ising_64 (pairing A).
  - RAPL package and RAPL core: refused by the trial-window rule, "the mean exceeded the maximum".
  - NVML: the harness marked it attributable (0.0373 J/ESS, idle σ 15.5 W, drift 12.7 W). **It is refused under v3** for these reasons:
    - The trial-window other-core rule fails for the whole trial: other-core mean 0.161 > 0.05, and core 4 is at 0.625 > 0.15. The rule is per trial, not per meter; the harness did not apply it to NVML (see the harness note below).
    - There is no adjacent THRML A or B trial, because the next bracket was refused and ABBA is incomplete.
    - The summary J/ESS is null (n = 1).
  - window_audit (n = 1) shows only sample_warmup_draws inside the window and gpu_ess_chain outside it, which complies with rule 1.
- Attempt 15: quiet preflight (max load1 0.09), lock held, but no report.json was ever written. The THRML arm failed to start in-window throughout:
  - 24,006 lines of "JAX has no CPU device" / "THRML serve said ''" from the `-cuda` venvs.
  - CUDA_ERROR_NO_DEVICE.
  - THRML_LIMIT oom SIGKILL at grid size.
  - Result: ungraded/unbracketed for all three meters.
- Parity at 777f0417 on jetson-hub, from the attempt-7 report: all seven levers pass the new design (per_site ≤ 0.0092 < 0.05). Under the old design, incremental fails at 0.050048828125 and is kept on record. That report records rustc 1.100.0.

### framework-128gb-0000, AMD RAPL (model-based)

- framework was reachable this time over the tailnet (100.91.48.80); the LAN address 192.168.4.47 timed out.
- quiet24 c82c1e96 exit.txt: `quiet_joules exit=0 grade=quiet attempt=39 host_lock_held=true` (2026-11-10T11:44Z). The driver stopped there and did not run again before the Nov 15 deadline.
- Attempts 1–38: load_gated at preflight. Attempts 1–29 have no lock field.
- Attempt 39 was graded quiet under the c82c1e96 harness: 600 trial-meter rows, 234 RAPL-package rows attributable, 15 workloads. **It is refused under regrade-v3**:
  - Rule 1: no window_audit, and the c82c1e96 metered window includes scoring and exact-law work (the same defect as 037). That cannot be subtracted from raw counter samples.
  - Rule 3: idle brackets are per A/B pair, not interleaved between trials.
- RAPL core: 0 attributable, with 247 refused for negative incremental energy, 47 inside the noise and 6 for drift.
- These old-rule joules are **not** v3 results and are not reported here.

## Harness note (for bmi-0)

In 777f0417 attempt 7, the trial-window other-core refusal was applied to the RAPL rows but not to the NVML row of the same trial. The NVML row was marked attributable even though the trial violated the trial-window rule. Under v3 this grading treats the rule as a trial property, so the row is refused.

## Files

- `receipt/<host>/<meter>-passed.csv`: header only (no passes).
- `receipt/<host>/<meter>-refusals.csv`: one row per attempt per meter, with reasons.
- `receipt/summary.json`
- `tools/grade_v3.py`: the grading script (deterministic over the copied logs).
- Source copies:
  - `m5/` (all report.json, events, loadavg, and macmon raw for the 42 energy-phase attempts)
  - `thor/`
  - `jetson-hub/`
  - `framework/` (excluding states.txt and thrml_raw.txt)
- `MANIFEST.sha256`: covers every file in this directory except itself.
