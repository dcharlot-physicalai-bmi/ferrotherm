# quiet preflight on 5c54005 (max load1 0.55); trial windows unguarded; NOT quiet-host joules under the c82c1e96 rule

Host `jetson-hub`, 13th Gen Intel Core i9-13900H, cpus `8-11`, `--device cpu`.
Harness `5c54005fd0ff8dc62adedda478faa52388911902`, `git_dirty` false. THRML
0.1.4, JAX 0.12.0.1, `device=cpu:0`. Attempt 3 ran
`2026-11-09T21:26:23.966342+00:00` to `2026-11-09T21:56:52.501316+00:00`
(4:26:23 PM to 4:56:52 PM EST). `exit.txt` says `0 grade=quiet attempt=3`.
That line, and `evidence_grade: quiet` inside `report.json`, are the 5c54005
stamp. They are not the label of this directory.

Three meters, kept apart. RAPL `package-0`, the RAPL `core` subdomain, and
NVML on one NVIDIA GeForce RTX 4050 Laptop GPU (`scope` `gpu`). Package joules,
core joules, and GPU joules are not added. No ratio of one meter to another,
or of ferrotherm to THRML, is computed here. This attempt is not pooled with
attempt 2 on this host, with `framework-128gb-0000`, or with the M5
`soc-telemetry` refusal.

The publish tarball sha256 is
`fecd65ee195bbc4bee7961b0495375fe52679d39cca7562796d46b8579c0b77d`.
`MANIFEST.txt` is the tarball's own list. Four raw dumps were left out; their
sha256s are in that file and the bytes are not reconstructed:

| left out | bytes | sha256 |
| --- | --- | --- |
| `attempt-003/thrml_raw.txt` | 283662008 | `d13eb5880fdbca739f3f285f72df0d401d03cb7201a95952508decdd4c06e36e` |
| `attempt-003-rapl-raw.jsonl` | 10980507 | `4d06cd848c95b07b5eb580bb06a630e56a58830e8e2d8e0fb7356c2e71bef519` |
| `attempt-001-rapl-raw.jsonl` | 413248 | `ed44516786e3b21d3c628334542d57ef8f4d1190ac46785d97bd045934f7dc80` |
| `attempt-002-rapl-raw.jsonl` | 405561 | `daaf1e9e0c81dae46b90d1bda7b2dca433ba043e103e2c0f90026224d456d2b8` |

They remain on the host under `evidence-5c54005-quiet24-jetsonhub/`.

## Why these are not quiet-host joules under c82c1e96

- 5c54005 has no trial-window contamination refusal. The receipt records `other_core_busy` outside the pinned CPUs 8–11 at up to 0.31 per trial, and `other_process_cpu_s` at up to 55.95 s. Under the c82c1e96 rule (mean ≤ 5%, no sample above 15%), some accepted trials would have refused.
- No host lock existed.
- 262 of 900 meter-trials were refused for bracket drift (max drift 35.86 W) and 2 for idle noise. `potts_path_q3_n6`, `tsp_iv4_three_body_d8`, and `ising_open_4x4` were fully refused on both sides.

The receipt's own figures for those sentences: the largest outside-core fraction is 0.3066, on cpu 2, `tsp_iv4_three_body_d8` THRML trial 5. The largest `other_process_cpu_s` is 55.95, on `potts_path_q3_n6` THRML trial 2. The largest `baseline_drift_w` is 35.8552919 W, on `potts_path_q3_n6` THRML trial 1, package. All 10 trials of each of those three workloads, both sides, all three meters, carry the drift refusal. The two idle-noise refusals are the note `attribution refused: incremental energy is inside the idle noise band`. 636 meter-trials are attributable. 262 + 2 + 636 = 900.

Preflight held. `quiet-gate.json`: `max_load1` 0.55, threshold 2, 60.018 s, `busy_readable` true. `phase_order` is energy then accuracy. `projected_prices_applied` is false.

## Post-hoc sensitivity, not a grade

`other_core_busy` is one fraction per core outside `8-11`, from `/proc/stat` at the start of the energy phase to `/proc/stat` after the pair. The same string is copied onto every meter row of the pair. There is no 1 Hz series, so there is no per-sample maximum. The c82c1e96 mean criterion cannot be evaluated. The c82c1e96 sample criterion cannot be evaluated.

`SENSITIVITY.md` compares those single cumulative fractions to 0.05 and 0.15. Of the 636 attributable meter-trials, 204 have a cross-core mean of that one snapshot above 0.05, and 138 have at least one outside core above 0.15. Package, core, and NVML share the busy string, so those counts are not three measurements. Later pairs include earlier pairs in the delta. Filtering on that comparison would be selection after the fact. It is not a grade, and it does not change a joule.

## Uncertainty

`UNCERTAINTY.md` copies every summary's budget: Student-t half-width, quantization, cadence and edge, bracket, and combined, separately for RAPL package, RAPL core, and NVML. Where no trial was attributable, `joules_per_ess` and `student_t_half` are null and the receipt still stores the other terms. The three fully refused workloads still have gross and incremental intervals in the summary; they are not attributable joules per effective sample.

On NVML rows the receipt's `idle_watts` is the RAPL package bracket for that pair, not a GPU idle. The NVML incremental is GPU gross minus that package idle times the row's seconds. That is what the file contains. The top-level `idle_watts` 3.79565197 (sigma 0.19049676) is the package baseline the energy object recorded. It is not a core baseline and not a GPU baseline.
