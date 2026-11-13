# Bracket drift, version 2

Superseded by rules v3, with interleaved idle in ABBA. Rules v3 scores outside the window for both sides, gives each meter its own idle bracket, and interleaves idle in ABBA. It is being implemented on `perf/large-grid`. The rule text is `bench/thrml-tsp/METHODOLOGY.md` on that branch, read here at `83dbb616cd9e5395f5f3d77f685fb146fc8a92d5`. Grid-scale trials already keep the score out of the joule window; see "What is measured, separately" in that file.

This note is not applied to attempt 037 or to any published grade. The 3σ test on that receipt stays as it ran. NVML for attempt 037 is ungraded: the traces have no raw NVML idle samples, so no `regrade-v3 (per-meter bracket)` was published beside the as-measured refusal.

Intel RAPL core-domain J/ESS on jetson-hub vs THRML 0.1.4 on JAX CPU, per workload, is the energy this note is about. Tarball sha256 1df27d9f6716ca6ad0da9a0fe4c773dfc8f5a46609e5b938cc086a00d7f1b613. Publish commit 22d3e720714c76e73389998d36e5b8159e89fe96. The drift numbers below are `baseline_drift_w` on the 300 trials of each meter in that receipt. They are not a new grade.

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

## Where the NVML bracket comes from

`sample_bracket` on jetson-hub takes the counter path. There is no wall meter, no INA238, and no tegrastats. `counter_slice_pair` reads RAPL package `energy_uj` and, beside it, the RAPL core counter. It takes four 0.25 s slices. It does not call NVML.

When the rows are stamped, a core row uses `core_watts` and `core_sigma`. A sub-rail row uses that rail's own series. Every other scope, including `nvml` / `gpu`, uses the package mean, the package sigma, and the package drift flag. `idle_before_w` and `idle_after_w` for those rows copy `BracketSample::watts`, which is the package.

The receipt matches that path on all 300 pairs. GPU `idle_before_w`, `idle_after_w`, `baseline_drift_w`, `idle_watts`, and `idle_sigma` equal the package row. Core `idle_before_w` differs on all 300 pairs.

The attempt-037 traces are chain energies and states (`*.energy.txt`, `*.states.txt`). `meters.json` lists RAPL domains. `events.jsonl` has no NVML samples. Each GPU trial stores one gross joule total for the work window. The raw NVML idle samples this proposal would test were never written down.

## Hypothetical NVML 3σ test

Hypothetical, and not a grade. A per-meter 3σ test needs the NVML samples from the idle brackets. This bundle does not contain them. There is no pass count and no fail count to report. The published NVML attribution stays refused, because the rows were judged on the package bracket.

Using the busy-window NVML counter as a stand-in for the idle bracket would answer a different question. This note does not do that.

## A rule that can be frozen before the next campaign

Do not raise the multiplier above 3 after seeing attempt 037. Before any later graded run, write down one of the following and leave it unchanged for that run.

1. Longer idle brackets. Choose the length on a calibration day that is not attempt 037, long enough that the within-bracket sigma includes the package wander the 1 s bracket misses. Refuse a trial when drift exceeds 3 times that longer bracket's sigma. Record the length before the first graded trial.
2. Interleaved idle. Take short idle slices inside the trial window and form the drift from those slices, still refused at 3σ, so the comparison and the work share a timescale. The slice length and the count are written down before the first graded trial.
3. A per-meter drift model fitted only on separate calibration windows: package, core, and an NVML idle bracket recorded from NVML. Freeze the coefficients, then refuse a graded trial when its drift exceeds 3 times the model's sigma for that meter. NVML uses its own bracket. It does not inherit package watts.
4. Each meter keeps its own idle-bracket series and its own 3σ test. Package, core, and NVML are read on the bracket itself. For NVML the samples are `nvmlDeviceGetTotalEnergyConsumption` over that bracket, converted to watts the same way the RAPL slices are, and `combine_bracket` runs on those watts and that sigma. A GPU row is refused when its own drift exceeds 3 times its own sigma, or when the NVML read produced fewer than two slices. Package drift does not set the GPU flag. Core already has this split. The fix is the fall-through that currently hands every non-core, non-subrail scope the package series. This rule can be frozen on its own, before the next graded run, whether or not rules 1–3 are also chosen.

Any one of these can be the predeclared rule. Mixing them after a graded trial has failed is a new campaign, not a correction of the old one.

## What would make it fail

A longer bracket whose sigma still does not cover a jump (sleep, a charger event, another job) stays refused. The multiplier is not raised to make that trial pass.

A calibration window that omits the THRML process, or that uses a different `--cpus` set than the graded run, does not transfer. The graded run is refused. It is not refit.

Rule 3 does not pass an NVML row that still has no bracket of its own.

Rule 4 fails closed when the NVML read is missing or short. The row is refused for having no NVML bracket. It is not filled from the package. A later NVML bracket whose own samples drift by more than 3σ stays refused. The multiplier is not raised, and the package result is not consulted to save it.

Applying any of this to attempt 037 would change a grade after the bracket was already observed. This proposal does not do that. Package and NVML on attempt 037 stay attribution-refused. Core stays attributable under the rule that ran. The hypothetical NVML test above has no samples, so it does not supply a substitute grade.
