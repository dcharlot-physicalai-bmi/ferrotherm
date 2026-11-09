# 0029c27

Revision `0029c270ea401109254f691ad78bc1666c9540df`. Both hosts are evidence grade **contended host**. 40 of 40 measured trials on each host were attributable. J/ESS is incremental joules per effective sample, per sampler repetition.

RAPL package and macmon `sys_power` are separate measurements. The tables below are not one comparison. `sys_power` is SoC telemetry (CPU, integrated GPU, ANE). It is not wall power. The original `report.json` files still say `scope: "whole-system"` on the macmon rows. That token is the harness label from this revision. It names `sys_power`. The bytes of those receipts were not rewritten.

Package and core stay separate. The 0029c27 harness did not emit a core column. The raw RAPL trace does contain `core_uj`, and that integral is reported on its own below. It is not added to package.

## Harness tables, as measured

These cells are copied from `TABLES.md`. They are the harness figures: RAPL is a counter delta, macmon is mean watts times duration. They were not replaced by a trapezoid. The intervals are the harness 95% Student-t scatter, n=5.

### RAPL package, `framework-128gb-0000` (primary)

| Workload | ferrotherm J/ESS | THRML J/ESS |
|---|---|---|
| thrml_readme_ising_chain_n5 | 4.73864417e-06 [4.45701112e-06, 5.02027723e-06] | 2.77753027e-03 [2.40850973e-03, 3.14655081e-03] |
| ising_grid2d_open_3x3 | 8.47697947e-06 [7.58536698e-06, 9.36859196e-06] | 4.48414457e-03 [4.42244895e-03, 4.54584019e-03] |
| potts_path_q3_n6 | 5.09328891e-06 [4.33088027e-06, 5.85569755e-06] | 7.65889751e-03 [7.07170181e-03, 8.24609322e-03] |
| tsp_iv4_three_body_d8 | 1.74431866e-05 [1.67304690e-05, 1.81559042e-05] | 2.97110511e-02 [2.02037979e-02, 3.92183044e-02] |

### macmon SoC telemetry (`sys_power`), M5 Max 48 GB (secondary)

| Workload | ferrotherm J/ESS | THRML J/ESS |
|---|---|---|
| thrml_readme_ising_chain_n5 | 4.40969033e-06 [4.25611040e-06, 4.56327027e-06] | 1.56413579e-03 [1.32624936e-03, 1.80202222e-03] |
| ising_grid2d_open_3x3 | 8.06818676e-06 [6.86359572e-06, 9.27277780e-06] | 2.80065807e-03 [2.47721379e-03, 3.12410234e-03] |
| potts_path_q3_n6 | 4.91204208e-06 [4.48560206e-06, 5.33848211e-06] | 5.79050226e-03 [5.44504263e-03, 6.13596188e-03] |
| tsp_iv4_three_body_d8 | 1.89290013e-05 [1.64946008e-05, 2.13634017e-05] | 2.30589743e-02 [1.75634105e-02, 2.85545381e-02] |

## Recomputation from the raw traces

Per-trial windows are not in the bundle. Each trial in `report.json` has `window_seconds` and no absolute start. `framework/evidence-0029c27/events.jsonl` marks the process start and end only. The M5 bundle has no events file. `run.log` has no wall-clock timestamps. The per-workload J/ESS cells were therefore not recomputed. Slicing the continuous traces by the published durations would be an estimate, and it was not done.

What the traces do contain:

### Framework RAPL, whole process, counter delta

`framework/evidence-0029c27/rapl-raw.jsonl`. Integer `package-0_uj` and `core_uj`, one sample about every 0.100 s, no wraps. The file tags each line `quiet` or `contended`. The integral is the sum of the recorded deltas from the second sample through the last. Edges are the first and last sample of that mode. This is the whole process, including the accuracy section. It is not a per-workload table, and idle was not subtracted.

| Mode | Samples | Span | Median cadence | Package J | Core J |
|---|---|---|---|---|---|
| quiet (harness outcome `load_gated`; not quiet-host energy) | 437 | 2026-11-07T12:27:43.337Z to 12:28:27.119Z (43.782 s) | 0.100296 s | 1835.457310 | 26.151541 |
| contended | 1057 | 2026-11-07T12:28:27.220Z to 12:30:13.152Z (105.932 s) | 0.100324 s | 5306.071820 | 180.844819 |

The events file agrees with those spans to a fraction of a second (`contended` process 12:28:27.220Z–12:30:13.253Z). The recorded quantum is 1 µJ, so both ends of a counter are 2e-6 J. On the contended package integral that is negligible next to 5306 J. One recorder interval of average power, the edge bound, is 5.025 J on contended package and 0.171 J on contended core. Those two scopes are not added.

### M5 macmon, whole recording, trapezoid of `sys_power`

`m5/evidence-0029c27/macmon-raw.jsonl`. 973 samples, 2026-11-07T12:27:43.032Z to 12:29:35.221Z, 112.188 s, median cadence 0.115612 s. `start.txt` is 07:27:42-05:00 and `end.txt` is 07:29:35-05:00, the same recording. There is no per-trial cut.

Both edges are on a sample because the integral runs from the first sample to the last. Trapezoid of `sys_power`: **2966.460208 J**. Mean of the same samples times duration: **2966.394379 J**. The difference is 0.065829 J, 2.22e-5 of the mean-times-duration figure. Sample count 973, cadence as above.

`sys_power` was unchanged on 841 of 972 steps (128 distinct values, 12.249 W to 36.309 W). `all_power` changed on every step (973 distinct values, 0.926 W to 25.284 W). Trapezoid of `all_power` over the same timestamps is 1132.887239 J. That field was not the one the harness integrated, and it is not a correction of the `sys_power` joules. It is why `sys_power` is not a wall measurement: a field that sits still while the other power fields move is not the plug.

macmon states no quantum. None was invented.

## Uncertainty budget on the published J/ESS

This budget sits beside the Student-t scatter. It does not replace the cells. Per trial, then averaged across the five trials and not divided by sqrt(5), because a shared instrument limit does not shrink when the trial is repeated:

- Student-t half-width: `(hi95 − lo95) / 2` of the published J/ESS summary.
- Quantization: `2 × 1e-6 J / repetitions / ess` for RAPL, from the integer microjoule counter in `rapl-raw.jsonl`. macmon states no quantum, so that term is omitted and is not entered as zero.
- Cadence and edge: one poll of the window's average power, `gross_per_rep / ((counter_updates − 1) × ess)`, using the trial's own window, sample count, gross, and ESS from `report.json`. The external 10 Hz RAPL recorder is not the harness poll (the harness recorded about 2000 updates in 2 s) and was not substituted for it.
- Bracket: `idle_sigma × seconds_per_rep / ess`, from the published baseline. This run did not record a before/after bracket, so there is no second drift term to add. The 3-sigma attribution rule is unchanged and is not this 1-sigma term.
- Combined: root-sum-square of the finite terms.
- Scope limit, not in the number: package excludes DRAM and the board, and does not include the core column; macmon `sys_power` is SoC telemetry, not wall power, and the stuck steps above are part of that limit.

| Scope | Workload | ferrotherm mean ± combined | THRML mean ± combined | THRML / ferrotherm | Intervals overlap |
|---|---|---|---|---|---|
| package | thrml_readme_ising_chain_n5 | 4.738644e-06 ± 3.150e-07 | 2.777530e-03 ± 3.730e-04 | 586 | no |
| package | ising_grid2d_open_3x3 | 8.476979e-06 ± 9.239e-07 | 4.484145e-03 ± 1.095e-04 | 529 | no |
| package | potts_path_q3_n6 | 5.093289e-06 ± 7.751e-07 | 7.658898e-03 ± 6.191e-04 | 1504 | no |
| package | tsp_iv4_three_body_d8 | 1.744319e-05 ± 8.599e-07 | 2.971105e-02 ± 9.529e-03 | 1703 | no |
| soc-telemetry | thrml_readme_ising_chain_n5 | 4.409690e-06 ± 4.715e-07 | 1.564136e-03 ± 2.891e-04 | 355 | no |
| soc-telemetry | ising_grid2d_open_3x3 | 8.068187e-06 ± 1.446e-06 | 2.800658e-03 ± 4.364e-04 | 347 | no |
| soc-telemetry | potts_path_q3_n6 | 4.912042e-06 ± 6.935e-07 | 5.790502e-03 ± 6.193e-04 | 1179 | no |
| soc-telemetry | tsp_iv4_three_body_d8 | 1.892900e-05 ± 3.188e-06 | 2.305897e-02 ± 5.910e-03 | 1218 | no |

The package column runs from 529× on `ising_grid2d_open_3x3` (open 3×3, 9 spins) to 1703× on `tsp_iv4_three_body_d8` (8 spins). The SoC-telemetry column runs from 347× on that same 3×3 to 1218× on that same 8-spin three-body model. The other two rows in each scope are the 5-spin chain (`thrml_readme_ising_chain_n5`, 586× package, 355× SoC telemetry) and the 6-site Potts path (`potts_path_q3_n6`, 1504× package, 1179× SoC telemetry). Those four models are the whole of this bundle. The span is not a 16×16, 32×32, or 64×64 result, and it is not a claim that ferrotherm costs less on a large grid. Every cell is grade contended host.

On those four models, and only those, ferrotherm J/ESS stays below THRML J/ESS after the combined budget, inside its own scope. The macmon ordering is an ordering of SoC-telemetry mean-times-duration figures. It is not a wall-power result, and it is not a per-trial trapezoidal recomputation: those windows are not in the trace. The whole-recording trapezoid of `sys_power` differs from mean-times-duration by 2.22e-5, which does not move the quotients in this table. The package ordering is an ordering of package counter deltas on the same four models. The core integral above is a different number and was not folded in.

## Limits

- Contended host. The Framework quiet attempt was `load_gated`. The M5 run was `--contended` and stays contended host. These figures are not quiet-host evidence.
- THRML 0.1.4 ran on JAX 0.12.0.1 CPU. There is no GPU THRML result here.
- The models are these four and no others: a 5-spin chain, a 3×3 Ising grid (9 spins), a 6-site Potts path, and an 8-spin three-body system. Each cell is five trials. The 529×–1703× package span and the 347×–1218× SoC-telemetry span are the endpoints of those four rows. They do not cover a larger grid.
- No quiet-host leadership claim, and no large-grid claim. This bundle does not say which sampler costs less on a quiet machine, on a 16×16 or larger lattice, or on a Z1 device.
