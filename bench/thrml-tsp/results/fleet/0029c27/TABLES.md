# Energy measurements — 0029c270

Values copied directly from the harness summaries, not recalculated. Cells are mean [95% CI]; n=5 per sampler per workload. Gross and above-baseline joules are normalized per sampler repetition, NOT totals for the >=2-second measurement window. J/ESS is incremental joules per effective sample. Exact report.json records and raw traces are included. Both measured runs have evidence grade **contended host**.

The macmon scope in these receipts was recorded as `whole-system`. That token is `sys_power`, SoC telemetry, not wall power. The cells below are the harness mean-times-duration figures. They were not replaced by a trapezoid: the raw traces have no per-trial timestamps. The whole-trace integrals, the uncertainty budget, and whether the ratios still separate are in `RESULTS.md`. Package and core are not added. This harness did not emit a core column.

## Framework — RAPL package, primary

Grade: contended host. Run-start load1=2.65; energy-section load1=[2.81]. Baseline=37.3700971 W, sigma=0.309415322 W. Attributable: 40/40. Window range: 2.000025–2.404664 s; counter updates: 1999–2397.

|Workload|Sampler|Gross J [95% CI]|Above-baseline J [95% CI]|J/ESS [95% CI]|Attributable|load1 range|
|---|---|---|---|---|---|---|
|thrml_readme_ising_chain_n5|ferrotherm|3.49753044e-02 [3.45073745e-02, 3.54432342e-02]|7.62297997e-03 [7.16986192e-03, 8.07609803e-03]|4.73864417e-06 [4.45701112e-06, 5.02027723e-06]|5/5|3.17–3.23|
|thrml_readme_ising_chain_n5|thrml|1.48533452e+01 [1.41554931e+01, 1.55511972e+01]|4.42705006e+00 [4.18247246e+00, 4.67162766e+00]|2.77753027e-03 [2.40850973e-03, 3.14655081e-03]|5/5|3.16–3.21|
|ising_grid2d_open_3x3|ferrotherm|4.22565478e-02 [4.14073796e-02, 4.31057160e-02]|9.51579834e-03 [8.51408589e-03, 1.05175108e-02]|8.47697947e-06 [7.58536698e-06, 9.36859196e-06]|5/5|3.11–3.16|
|ising_grid2d_open_3x3|thrml|1.69857013e+01 [1.66777909e+01, 1.72936117e+01]|4.95333738e+00 [4.82329748e+00, 5.08337727e+00]|4.48414457e-03 [4.42244895e-03, 4.54584019e-03]|5/5|3.11–3.15|
|potts_path_q3_n6|ferrotherm|3.83715276e-02 [3.72722332e-02, 3.94708219e-02]|8.91746875e-03 [7.58209876e-03, 1.02528387e-02]|5.09328891e-06 [4.33088027e-06, 5.85569755e-06]|5/5|3.09–3.16|
|potts_path_q3_n6|thrml|5.70122906e+01 [5.59564115e+01, 5.80681697e+01]|1.39607768e+01 [1.30233807e+01, 1.48981729e+01]|7.65889751e-03 [7.07170181e-03, 8.24609322e-03]|5/5|3.09–3.16|
|tsp_iv4_three_body_d8|ferrotherm|3.89789301e-02 [3.79511372e-02, 4.00067230e-02]|9.01661957e-03 [8.65248664e-03, 9.38075249e-03]|1.74431866e-05 [1.67304690e-05, 1.81559042e-05]|5/5|3.12–3.19|
|tsp_iv4_three_body_d8|thrml|3.88973009e+01 [3.75147609e+01, 4.02798408e+01]|1.07228736e+01 [1.05511626e+01, 1.08945846e+01]|2.97110511e-02 [2.02037979e-02, 3.92183044e-02]|5/5|3.12–3.34|

Quiet attempt: load_gated; load samples [2.86, 2.79, 2.79, 2.79, 2.73, 2.73, 2.75, 2.75, 2.75, 2.77, 2.77, 2.71, 2.71, 2.71, 2.65]. No quiet-host energy trials.

## M5 Max 48GB — macmon SoC telemetry (`sys_power`), secondary

Grade: contended host. Run-start load1=1.63; energy-section load1=[1.83]. Baseline=12.8485681 W, sigma=0.153944291 W. Attributable: 40/40. Window range: 2.000041–2.713468 s; counter updates: 18–25.

|Workload|Sampler|Gross J [95% CI]|Above-baseline J [95% CI]|J/ESS [95% CI]|Attributable|load1 range|
|---|---|---|---|---|---|---|
|thrml_readme_ising_chain_n5|ferrotherm|1.25575740e-02 [1.23448790e-02, 1.27702689e-02]|7.09339689e-03 [6.84585859e-03, 7.34093518e-03]|4.40969033e-06 [4.25611040e-06, 4.56327027e-06]|5/5|2.49–2.75|
|thrml_readme_ising_chain_n5|thrml|4.99957169e+00 [4.64080081e+00, 5.35834257e+00]|2.51641786e+00 [2.18592886e+00, 2.84690687e+00]|1.56413579e-03 [1.32624936e-03, 1.80202222e-03]|5/5|2.64–2.75|
|ising_grid2d_open_3x3|ferrotherm|1.59285411e-02 [1.46396927e-02, 1.72173894e-02]|9.06510956e-03 [7.71314475e-03, 1.04170744e-02]|8.06818676e-06 [6.86359572e-06, 9.27277780e-06]|5/5|2.67–2.79|
|ising_grid2d_open_3x3|thrml|5.96819473e+00 [5.67721419e+00, 6.25917527e+00]|3.05959324e+00 [2.69712091e+00, 3.42206556e+00]|2.80065807e-03 [2.47721379e-03, 3.12410234e-03]|5/5|2.67–2.87|
|potts_path_q3_n6|ferrotherm|1.69482621e-02 [1.62016368e-02, 1.76948874e-02]|8.60117667e-03 [7.85559190e-03, 9.34676144e-03]|4.91204208e-06 [4.48560206e-06, 5.33848211e-06]|5/5|2.82–2.96|
|potts_path_q3_n6|thrml|2.22324002e+01 [2.19661161e+01, 2.24986844e+01]|1.06432397e+01 [1.03461985e+01, 1.09402809e+01]|5.79050226e-03 [5.44504263e-03, 6.13596188e-03]|5/5|2.76–2.96|
|tsp_iv4_three_body_d8|ferrotherm|1.82445555e-02 [1.70795419e-02, 1.94095692e-02]|9.79340246e-03 [8.54508117e-03, 1.10417238e-02]|1.89290013e-05 [1.64946008e-05, 2.13634017e-05]|5/5|2.61–2.76|
|tsp_iv4_three_body_d8|thrml|1.58536676e+01 [1.48880262e+01, 1.68193090e+01]|7.79886450e+00 [6.82056854e+00, 8.77716046e+00]|2.30589743e-02 [1.75634105e-02, 2.85545381e-02]|5/5|2.48–2.72|

