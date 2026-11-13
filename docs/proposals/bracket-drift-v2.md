# Bracket drift, version 2

Proposal only. It is not applied to attempt 037 or to any published grade. The 3σ test stays as it ran.

Intel RAPL core-domain J/ESS on jetson-hub vs THRML 0.1.4 on JAX CPU, per workload, is the energy this note is about. Tarball sha256 1df27d9f6716ca6ad0da9a0fe4c773dfc8f5a46609e5b938cc086a00d7f1b613. Publish commit PUBLISH_SHA_PLACEHOLDER. The drift numbers below are `baseline_drift_w` on the 300 trials of each meter in that receipt. They are not a new grade.

## Measured drift

n = 300 trials per meter, both sides and all 15 workloads.

| meter | min W | median W | mean W | p90 W | p99 W | max W | trials with drift > 3 × idle σ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Intel RAPL package | 1.438 | 1.975 | 1.967 | 2.246 | 2.499 | 2.592 | 300 |
| NVML RTX 4050 | 1.438 | 1.975 | 1.967 | 2.246 | 2.499 | 2.592 | 300 |
| Intel RAPL core | 0.0003 | 0.050 | 0.059 | 0.115 | 0.223 | 0.249 | 0 |

Package idle-bracket sigma on those 300 trials ran from 0.086 W to 0.728 W (mean 0.323 W). One package trial, ferrotherm `thrml_readme_ising_chain_n5` trial 0, recorded idle before 3.462 W, idle after 5.337 W, drift 1.875 W, idle sigma 0.299 W. Three sigma of that bracket is 0.897 W, so 1.875 W fails. The NVML series is identical to the package series, and `idle_before_w` matches the package row on all 300 pairs. Core drift never reached 0.5 W, and 0 of 300 core trials failed 3σ. All 300 core trials stayed attributable.

## Cause

The test compares the absolute difference of two idle brackets with 3 times the standard deviation inside a bracket. On the package, the move from one bracket to the next is about 2 W, and the scatter inside a bracket is about 0.3 W. A 1 s bracket does not contain that slower move, so every package trial fails. The core domain moves by tens of milliwatts and passes the same test.

NVML rows in this receipt store the package idle bracket. Their refusal is that package test. It is not a separate GPU baseline.

## A rule that can be frozen before the next campaign

Do not raise the multiplier above 3 after seeing attempt 037. Before any later graded run, write down one of the following and leave it unchanged for that run.

1. Longer idle brackets. Choose the length on a calibration day that is not attempt 037, long enough that the within-bracket sigma includes the package wander the 1 s bracket misses. Refuse a trial when drift exceeds 3 times that longer bracket's sigma. Record the length before the first graded trial.
2. Interleaved idle. Take short idle slices inside the trial window and form the drift from those slices, still refused at 3σ, so the comparison and the work share a timescale. The slice length and the count are written down before the first graded trial.
3. A per-meter drift model fitted only on separate calibration windows: package, core, and an NVML idle bracket recorded from NVML. Freeze the coefficients, then refuse a graded trial when its drift exceeds 3 times the model's sigma for that meter. NVML uses its own bracket. It does not inherit package watts.

Any one of these can be the predeclared rule. Mixing them after a graded trial has failed is a new campaign, not a correction of the old one.

## What would make it fail

A longer bracket whose sigma still does not cover a jump (sleep, a charger event, another job) stays refused. The multiplier is not raised to make that trial pass.

A calibration window that omits the THRML process, or that uses a different `--cpus` set than the graded run, does not transfer. The graded run is refused. It is not refit.

Rule 3 does not pass an NVML row that still has no bracket of its own.

Applying any of this to attempt 037 would change a grade after the bracket was already observed. This proposal does not do that. Package and NVML on attempt 037 stay attribution-refused. Core stays attributable under the rule that ran.
