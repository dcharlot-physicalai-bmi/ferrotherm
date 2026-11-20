# Energy methodology v3.1

This is the v3.1 rule. It does not edit `bench/thrml-tsp/METHODOLOGY.md`. That
file stays the frozen v3 text, commit
`dbf9653a6091543df420f9ab8a3fe5da9b00a758`. Brackets, the 3σ test, ABBA order,
the load1 settle, the 0.05 parity line, and the stop rule are unchanged.
Their definitions are the v3 text.

## Base

`pre_rewrite_sha` is `777f0417340788428cf0525003277f9829ca6e3a`. The tree of
this branch's base, `ffe89b0702553b1c70dbbe1e52b8c537d6c0bd0b`
(`rewrite/perf/large-grid`), is that commit. The defect, the frozen v3 text,
and the attempt-007 evidence are `pre_rewrite_sha`. v3.1 does not regrade it.

## The trial window is a property of the trial

The trial-window other-core rule applies identically to every meter row of a
trial. The rows are RAPL package, RAPL core, NVML, macmon, and ina238 or
tegrastats. A meter read on its own counter beside the watched window takes
the same refusal as the rows the window stamped. It does not keep a grade
the window refused.

The thresholds are the v3 thresholds. On Linux the signal is the 1 Hz busy
fraction of cores outside `--cpus`. The receipt stamps `other_core_mean_max`
0.05 and `other_core_sample_max` 0.15. A trial is refused when the mean is
above the first or any sample is above the second, or when the window
recorded no finite sample. The fraction is not subtracted from joules.

On macOS the signal is still the non-harness CPU fraction, mean at most 0.15
and no sample above 0.50, and the idle brackets still have to agree within
3σ. That answer is written onto every meter row of the trial as well. v3.1
does not add a threshold and does not change which operating system uses
which signal.

`load1` is still not a trial-window criterion. Brackets still use it.

## Why

`pre_rewrite_sha` `777f0417340788428cf0525003277f9829ca6e3a` computes the
refusal inside the watched window and copies it onto the rows that window
measured. On a host with RAPL and NVML, the package row and the core row are
inside that window. The NVML row is a second counter pass and was left
without the refusal.

Measured on jetson-hub, native v3 attempt 7 (attempt-007), that harness, one
ferrotherm-gpu trial of `grid_scale_ising_64`, pairing A. The hand grade of
2026-11-20 read the trial as follows. RAPL package and RAPL core were
refused by the trial-window rule, reason `the mean exceeded the maximum`.
The same trial's other-core mean was 0.161, above 0.05, and core 4 was at
0.625, above 0.15. The harness marked the NVML row attributable at 0.0373
J/ESS, with idle σ 15.5 W and drift 12.7 W. Those joules are the hand
grade's reading of the receipt. They are not a new measurement.

The hand grade refused that NVML row under v3, because the rule is a property
of the trial. The harness had not applied it to the NVML row.

## Receipts

A receipt written by the v3.1 harness carries `rules_version` `v3.1`.

## Already graded

Joules collected under `pre_rewrite_sha`
`777f0417340788428cf0525003277f9829ca6e3a` and graded under v3 stay graded
under v3, by hand. v3.1 does not regrade them. The attempt-007 NVML row
remains that hand refusal. It is not a v3.1 output.
