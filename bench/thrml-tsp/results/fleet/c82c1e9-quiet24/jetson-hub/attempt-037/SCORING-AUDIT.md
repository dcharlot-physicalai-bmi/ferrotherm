# Scoring inside the attempt-037 energy window

Intel RAPL core-domain J/ESS on jetson-hub vs THRML 0.1.4 on JAX CPU, per workload. Tarball sha256 1df27d9f6716ca6ad0da9a0fe4c773dfc8f5a46609e5b938cc086a00d7f1b613. Publish commit 22d3e720714c76e73389998d36e5b8159e89fe96.

This audit does not change a grade and does not replace a published J/ESS. The published figures stay the receipt means.

## What the two timers cover

The energy window calls `repeat_until_window` inside `sample_during`. Every repetition of that loop is on the meter.

Ferrotherm calls `ft::run`. The sampler timer in `run_ising`, `run_potts`, and `run_hubo` covers `collect` or the heat-bath sweeps. Graph and model construction run before that timer. Scoring runs after it. Both are inside the energy window. The accuracy-phase `seconds` are that sampler timer.

THRML calls `ThrmlServe::run`. Python's `perf_counter` covers `sample_states` and `block_until_ready`. Decode and the state print run after that timer, still inside the energy window. Rust then parses the states and calls `score_observed`, which rebuilds the model, evaluates the energy of every returned state, and scores. The accuracy-phase THRML `seconds` are the Python sampler timer. Rust scoring of the accuracy phase happens later, outside that phase's Python process, and is not the energy-window timer.

`scaling()` builds the transfer-matrix means and the 32×32 and 64×64 reference-chain means once, at process start, before the energy phase. A workload whose `exact_energy` is that stored mean uses `score_mean_only` inside the window: Geyer's initial monotone sequence on the energy trace. It does not rebuild the transfer matrix or the reference chain per repetition.

## What runs inside a repetition

| workload | law | inside each ferrotherm repetition | inside each THRML repetition |
| --- | --- | --- | --- |
| `thrml_readme_ising_chain_n5` | 2^5 = 32 | `exact_boltzmann`, then the mean and the histogram | same enumeration after the state return |
| `ising_grid2d_open_3x3` | 2^9 = 512 | `exact_boltzmann` | same enumeration after the state return |
| `potts_path_q3_n6` | 3^6 = 729 | `potts::enumerate` | same enumeration after the state return |
| `tsp_iv4_three_body_d8` | 2^8 = 256 | `exact_hubo` and a second walk for the mean | `exact_hubo` after the state return |
| `ising_open_4x4` | 2^16 = 65536 | `exact_boltzmann` | same enumeration after the state return |
| `tsp_iv4_three_body_d12` | 2^12 = 4096 | `exact_hubo` | `exact_hubo` after the state return |
| `tsp_iv4_three_body_d16` | 2^16 = 65536 | `exact_hubo` | `exact_hubo` after the state return |
| `ising_open_8x8`, `ising_open_16x16` | transfer-matrix mean, stored | `score_mean_only` | energy of each returned state, then `score_mean_only` |
| `ising_open_32x32`, `ising_open_64x64` | reference-chain mean, stored | `score_mean_only` | energy of each returned state, then `score_mean_only` |
| `potts_open_q3_4x4`, `potts_open_q3_8x8`, `potts_open_q4_4x4`, `potts_open_q4_8x8` | transfer-matrix mean, stored | `score_mean_only` | energy of each returned state, then `score_mean_only` |

The three-body models have no separate tour check in this window. Their exact computation is `exact_hubo` over 2^d.

The 4×4 transfer-matrix cross-check against enumeration also runs inside `ising_square`, once, at process start. The per-repetition cost is the later `exact_boltzmann`, because `exact_energy` stays unset for side ≤ 4.

## Timer comparison

Sampler time is the mean of the 10 accuracy-phase trials. Metered time is the mean of `joules.seconds` on energy trials 1–9, Intel RAPL core. Trial 0 is left out of the ratio so the one-time THRML start, below, does not enter the scoring estimate. Ferrotherm trial 0 matches the other nine.

The ratio is metered time divided by sampler time. ESS comes from the chain and does not grow when the score runs longer, so under constant incremental core power the published J/ESS is high by that ratio. The sampling-only column is the published mean multiplied by sampler/metered. It is an estimate under that power assumption. Incremental joules are one number for the whole repetition, so a hotter or colder score moves the joule share off the time share. Rows whose ratio sits with the stored-mean ferrotherm cluster, 0.97 to 1.04, have no resolved excess. A negative share is that cluster, not a credit.

| workload | side | sampler s | metered s | ratio | excess share | published core J/ESS | sampling-only estimate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `thrml_readme_ising_chain_n5` | ferrotherm | 5.173e-4 | 7.177e-4 | 1.387 | 27.9% | 8.73849531e-6 | 6.30e-6 |
| `thrml_readme_ising_chain_n5` | thrml | 3.151e-1 | 3.204e-1 | 1.017 | 1.7% | 7.65167795e-3 | 7.52e-3 |
| `ising_grid2d_open_3x3` | ferrotherm | 6.290e-4 | 8.594e-4 | 1.366 | 26.8% | 1.50709324e-5 | 1.10e-5 |
| `ising_grid2d_open_3x3` | thrml | 3.611e-1 | 3.782e-1 | 1.047 | 4.5% | 8.07703124e-3 | 7.71e-3 |
| `potts_path_q3_n6` | ferrotherm | 6.383e-4 | 8.501e-4 | 1.332 | 24.9% | 1.01118403e-5 | 7.59e-6 |
| `potts_path_q3_n6` | thrml | 1.242e0 | 1.275e0 | 1.027 | 2.6% | 1.52492689e-2 | 1.49e-2 |
| `tsp_iv4_three_body_d8` | ferrotherm | 6.120e-4 | 7.911e-4 | 1.293 | 22.6% | 2.99650630e-5 | 2.32e-5 |
| `tsp_iv4_three_body_d8` | thrml | 8.375e-1 | 8.608e-1 | 1.028 | 2.7% | 4.80239546e-2 | 4.67e-2 |
| `ising_open_4x4` | ferrotherm | 8.843e-4 | 7.933e-3 | 8.971 | 88.9% | 1.22954420e-4 | 1.37e-5 |
| `ising_open_4x4` | thrml | 3.601e-1 | 3.886e-1 | 1.079 | 7.3% | 6.67543119e-3 | 6.19e-3 |
| `ising_open_8x8` | ferrotherm | 2.586e-3 | 2.675e-3 | 1.034 | 3.3% | 5.19702797e-5 | 5.02e-5 |
| `ising_open_8x8` | thrml | 3.817e-1 | 4.019e-1 | 1.053 | 5.0% | 8.24619319e-3 | 7.83e-3 |
| `ising_open_16x16` | ferrotherm | 9.773e-3 | 9.560e-3 | 0.978 | unresolved | 1.98434699e-4 | — |
| `ising_open_16x16` | thrml | 3.506e-1 | 4.137e-1 | 1.180 | 15.2% | 9.67753843e-3 | 8.20e-3 |
| `ising_open_32x32` | ferrotherm | 3.888e-2 | 3.764e-2 | 0.968 | unresolved | 8.13311203e-4 | — |
| `ising_open_32x32` | thrml | 3.919e-1 | 5.773e-1 | 1.473 | 32.1% | 1.38782810e-2 | 9.42e-3 |
| `ising_open_64x64` | ferrotherm | 1.545e-1 | 1.504e-1 | 0.973 | unresolved | 3.28229284e-3 | — |
| `ising_open_64x64` | thrml | 3.975e-1 | 1.096e0 | 2.757 | 63.7% | 2.93838190e-2 | 1.07e-2 |
| `potts_open_q3_4x4` | ferrotherm | 2.386e-3 | 2.479e-3 | 1.039 | 3.8% | 3.25698304e-5 | 3.13e-5 |
| `potts_open_q3_4x4` | thrml | 5.408e-1 | 5.761e-1 | 1.065 | 6.1% | 7.11206083e-3 | 6.68e-3 |
| `potts_open_q3_8x8` | ferrotherm | 9.710e-3 | 9.438e-3 | 0.972 | unresolved | 1.27207981e-4 | — |
| `potts_open_q3_8x8` | thrml | 5.787e-1 | 6.122e-1 | 1.058 | 5.5% | 8.06332649e-3 | 7.62e-3 |
| `potts_open_q4_4x4` | ferrotherm | 3.096e-3 | 3.133e-3 | 1.012 | 1.2% | 3.84741626e-5 | 3.80e-5 |
| `potts_open_q4_4x4` | thrml | 6.908e-1 | 7.337e-1 | 1.062 | 5.8% | 9.28341410e-3 | 8.74e-3 |
| `potts_open_q4_8x8` | ferrotherm | 1.319e-2 | 1.286e-2 | 0.975 | unresolved | 1.63414243e-4 | — |
| `potts_open_q4_8x8` | thrml | 7.267e-1 | 7.580e-1 | 1.043 | 4.1% | 9.64319856e-3 | 9.24e-3 |
| `tsp_iv4_three_body_d12` | ferrotherm | 9.357e-4 | 1.683e-3 | 1.799 | 44.4% | 2.16495705e-4 | 1.20e-4 |
| `tsp_iv4_three_body_d12` | thrml | 1.112e0 | 1.174e0 | 1.056 | 5.3% | 2.29838563e-1 | 2.18e-1 |
| `tsp_iv4_three_body_d16` | ferrotherm | 1.116e-3 | 1.320e-2 | 11.828 | 91.5% | 5.22169163e-4 | 4.41e-5 |
| `tsp_iv4_three_body_d16` | thrml | 1.167e0 | 1.237e0 | 1.060 | 5.6% | 5.93470651e-2 | 5.60e-2 |

## Affected rows

A row is affected when the ratio is at least 1.15. That cut sits above the ferrotherm stored-mean cluster (0.97–1.04) and above the THRML stored-mean cluster (1.04–1.07), which is decode, the state print, and one energy evaluation per draw.

Ferrotherm, enumeration inside the window:

| workload | states enumerated | ratio | share of published J/ESS | published | sampling-only estimate |
| --- | --- | --- | --- | --- | --- |
| `thrml_readme_ising_chain_n5` | 32 | 1.387 | 28% | 8.73849531e-6 | 6.30e-6 |
| `ising_grid2d_open_3x3` | 512 | 1.366 | 27% | 1.50709324e-5 | 1.10e-5 |
| `potts_path_q3_n6` | 729 | 1.332 | 25% | 1.01118403e-5 | 7.59e-6 |
| `tsp_iv4_three_body_d8` | 256 | 1.293 | 23% | 2.99650630e-5 | 2.32e-5 |
| `tsp_iv4_three_body_d12` | 4096 | 1.799 | 44% | 2.16495705e-4 | 1.20e-4 |
| `ising_open_4x4` | 65536 | 8.971 | 89% | 1.22954420e-4 | 1.37e-5 |
| `tsp_iv4_three_body_d16` | 65536 | 11.828 | 92% | 5.22169163e-4 | 4.41e-5 |

`ising_open_4x4` stays under investigation. The estimate says about 89% of its published ferrotherm core J/ESS is the excess over the sampler timer. The sampling-only estimate, 1.37e-5, lies below the open 8×8 published mean 5.19702797e-5. That comparison is the constant-power estimate. It is not a revised result.

THRML, state return and per-state energy, then `score_mean_only`. These three do not re-run a transfer matrix or a reference chain:

| workload | returned record | ratio | share of published J/ESS | published | sampling-only estimate |
| --- | --- | --- | --- | --- | --- |
| `ising_open_16x16` | 2000 states × 256 spins | 1.180 | 15% | 9.67753843e-3 | 8.20e-3 |
| `ising_open_32x32` | 2000 × 1024 | 1.473 | 32% | 1.38782810e-2 | 9.42e-3 |
| `ising_open_64x64` | 2000 × 4096 | 2.757 | 64% | 2.93838190e-2 | 1.07e-2 |

The open 64×64 THRML core J/ESS is the published 2.93838190e-2. Under the constant-power estimate, about 64% of that figure is the return path. The sampling-only estimate is 1.07e-2. The published quotient against ferrotherm, 8.952 ± 0.762, uses the published means. This audit does not restate it as a sampling-only quotient.

Enumeration also runs on the THRML side of the seven ferrotherm rows above. On those rows the JAX sample is 0.32 s to 1.24 s, and the ratio stays at 1.02–1.08, inside the stored-mean THRML cluster. The ferrotherm excess on the 4×4 is 7.0 ms and on d=16 is 12.1 ms. That is a few percent of the THRML repetition, and these timers do not separate it from decode and the state print.

## One THRML start, not a score

`ThrmlSlot::arm` starts the Python process on the first THRML call, inside that trial's meter. The runner JIT-compiles every workload before it answers `READY`. That landed in `thrml_readme_ising_chain_n5` THRML trial 0: window 12.295 s and 6 repetitions, against about 2.22 s and 7 repetitions on each later trial. Trial 0 core J/ESS is 3.578e-2. The mean of trials 1–9 is 4.526e-3. The published 10-trial mean is 7.65167795e-3. The scoring ratio in the table excludes trial 0. The published mean still includes it.
