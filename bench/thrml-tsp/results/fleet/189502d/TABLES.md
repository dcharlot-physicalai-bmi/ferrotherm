# Measured energy at 189502d

All values below are derived from the unmodified report.json trial records. Mean [two-sided 95% Student-t CI], n=5 per cell. The t critical value is 2.7764451052, matching the pinned harness. Workload IDs were reconstructed from canonical workload order and the fixed 10-row A/B schedule per workload; assertions checked side/trial sequence. The harness’s own summaries pool workloads and are not used here. No trial was removed and negative values/interval bounds are retained.

F=ferrotherm; T=THRML. J means joules. J/ESS means incremental joules per effective sample, as recorded by the harness.

## Framework RAPL (primary)

Evidence grade: contended host. Energy-section load1=[2.77]; run-start load1=2.41. Baseline=37.4752469 W; baseline sigma=0.200209929 W.

|Workload|Side|Gross J [95% CI]|Above-baseline J [95% CI]|J/ESS [95% CI]|trial load1 range|
|---|---|---|---|---|---|
|thrml_readme_ising_chain_n5|F|3.75772000e-02 [1.11487231e-02, 6.40056769e-02]|5.08657315e-03 [-2.12478658e-02, 3.14210121e-02]|4.42756512e-06 [-1.23037485e-05, 2.11588787e-05]|2.87–2.87|
|thrml_readme_ising_chain_n5|T|1.61564032e+01 [1.20046413e+01, 2.03081651e+01]|4.63781093e+00 [3.58565998e+00, 5.68996187e+00]|3.67015719e-03 [1.65449365e-03, 5.68582073e-03]|2.87–2.87|
|ising_grid2d_open_3x3|F|4.87066000e-02 [4.48338445e-02, 5.25793555e-02]|1.47774333e-02 [1.15306245e-02, 1.80242421e-02]|1.33429540e-05 [1.01750089e-05, 1.65108991e-05]|2.87–2.87|
|ising_grid2d_open_3x3|T|1.69911898e+01 [1.61916536e+01, 1.77907260e+01]|4.69009687e+00 [4.59737669e+00, 4.78281706e+00]|4.11040968e-03 [3.62718097e-03, 4.59363840e-03]|2.87–2.87|
|potts_path_q3_n6|F|4.76630000e-02 [4.47383399e-02, 5.05876601e-02]|1.66724773e-02 [1.35688835e-02, 1.97760711e-02]|1.06423344e-05 [7.02435367e-06, 1.42603152e-05]|2.87–2.88|
|potts_path_q3_n6|T|5.58773288e+01 [5.46915367e+01, 5.70631209e+01]|1.21788175e+01 [1.12236189e+01, 1.31340161e+01]|6.79003211e-03 [5.79442895e-03, 7.78563527e-03]|2.87–2.88|
|tsp_iv4_three_body_d8|F|4.64084000e-02 [4.32895256e-02, 4.95272744e-02]|1.45117408e-02 [7.67064542e-03, 2.13528362e-02]|1.04067873e-04 [4.95079356e-06, 2.03184953e-04]|2.88–2.89|
|tsp_iv4_three_body_d8|T|3.77725342e+01 [3.67133321e+01, 3.88317363e+01]|9.54709071e+00 [9.03915946e+00, 1.00550220e+01]|3.64941231e-02 [-8.85809156e-03, 8.18463377e-02]|2.89–2.89|

## M5 Max macmon (secondary)

Evidence grade: contended host. Energy-section load1=[2.34]; run-start load1=2.27. Baseline=16.7105363 W; baseline sigma=1.83278712 W.

|Workload|Side|Gross J [95% CI]|Above-baseline J [95% CI]|J/ESS [95% CI]|trial load1 range|
|---|---|---|---|---|---|
|thrml_readme_ising_chain_n5|F|5.33680392e+01 [5.10315405e+01, 5.57045380e+01]|2.82984601e+01 [2.59615914e+01, 3.06353288e+01]|5.30594393e-06 [4.51270579e-06, 6.09918206e-06]|3.04–3.25|
|thrml_readme_ising_chain_n5|T|5.32845483e+01 [4.68569898e+01, 5.97121068e+01]|2.70137431e+01 [2.13530349e+01, 3.26744514e+01]|2.65982117e-03 [1.25599517e-03, 4.06364716e-03]|3.11–3.25|
|ising_grid2d_open_3x3|F|5.20338764e+01 [5.02410605e+01, 5.38266923e+01]|2.69638329e+01 [2.51694332e+01, 2.87582326e+01]|8.43747327e-06 [7.36102007e-06, 9.51392646e-06]|3.21–3.27|
|ising_grid2d_open_3x3|T|5.27503220e+01 [4.98469189e+01, 5.56537250e+01]|2.59214103e+01 [2.31843833e+01, 2.86584372e+01]|3.24565013e-03 [2.72740960e-03, 3.76389066e-03]|3.21–3.27|
|potts_path_q3_n6|F|4.69575648e+01 [4.32800644e+01, 5.06350651e+01]|2.18854145e+01 [1.82051845e+01, 2.55656445e+01]|5.05804580e-06 [3.40838558e-06, 6.70770602e-06]|3.25–3.53|
|potts_path_q3_n6|T|5.46020453e+01 [5.12667446e+01, 5.79373459e+01]|2.40202630e+01 [2.02146778e+01, 2.78258482e+01]|6.70502424e-03 [5.27842761e-03, 8.13162087e-03]|3.31–3.53|
|tsp_iv4_three_body_d8|F|4.48501586e+01 [4.32439290e+01, 4.64563882e+01]|1.97805291e+01 [1.81741960e+01, 2.13868622e+01]|5.13720502e-05 [5.83707982e-06, 9.69070206e-05]|3.07–3.37|
|tsp_iv4_three_body_d8|T|5.29209750e+01 [5.12848772e+01, 5.45570729e+01]|2.05973181e+01 [1.92460689e+01, 2.19485673e+01]|2.71081406e-02 [-8.49938930e-03, 6.27156706e-02]|3.07–3.37|

