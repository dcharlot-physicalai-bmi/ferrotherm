# Uncertainty budget

Intel RAPL core-domain J/ESS on jetson-hub vs THRML 0.1.4 on JAX CPU, per workload. Tarball sha256 1df27d9f6716ca6ad0da9a0fe4c773dfc8f5a46609e5b938cc086a00d7f1b613. Publish commit 22d3e720714c76e73389998d36e5b8159e89fe96.

These are the small accuracy-suite workloads, n=5 through the open 64×64 (n=4096). This is not a large-grid energy claim. Package energy and NVML energy are not J/ESS and are not in the quotient table.

Combined uncertainty is the root-sum-square of the receipt's Student-t half-width, quantization, cadence/edge, and bracket terms. The quotient is THRML core J/ESS divided by ferrotherm core J/ESS. Its uncertainty is `R * sqrt((u_t/J_t)^2 + (u_f/J_f)^2)` using those combined terms. `ising_open_4x4` is under investigation and is not cited as a result.

On the open 64×64, the core-domain quotient is 8.952 ± 0.762, about 9×. That row is this lattice only.

## Intel RAPL core

| workload | side size | ferrotherm J/ESS ± combined | THRML J/ESS ± combined | THRML / ferrotherm | status |
| --- | --- | --- | --- | --- | --- |
| thrml_readme_ising_chain_n5 | n=5, open chain | 8.73849531e-6 ± 3.35352066e-7 | 7.65167795e-3 ± 7.07118807e-3 | 875.6 ± 809.9 | reported |
| ising_grid2d_open_3x3 | n=9, open 3×3 | 1.50709324e-5 ± 4.60134697e-7 | 8.07703124e-3 ± 2.36856071e-4 | 535.9 ± 22.69 | reported |
| potts_path_q3_n6 | n=6, q=3, Potts path | 1.01118403e-5 ± 2.43255691e-7 | 1.52492689e-2 ± 1.03631757e-3 | 1508 ± 108.7 | reported |
| tsp_iv4_three_body_d8 | n=8, three-body d8 | 2.99650630e-5 ± 7.74024674e-7 | 4.80239546e-2 ± 1.61948198e-2 | 1603 ± 542.0 | reported |
| ising_open_4x4 | n=16, open 4x4 | 1.22954420e-4 ± 3.25039702e-6 | 6.67543119e-3 ± 2.31815631e-4 | under investigation | under investigation |
| ising_open_8x8 | n=64, open 8x8 | 5.19702797e-5 ± 1.43952957e-6 | 8.24619319e-3 ± 2.32210688e-4 | 158.7 ± 6.267 | reported |
| ising_open_16x16 | n=256, open 16x16 | 1.98434699e-4 ± 4.12318572e-6 | 9.67753843e-3 ± 3.49653366e-4 | 48.77 ± 2.033 | reported |
| ising_open_32x32 | n=1024, open 32x32 | 8.13311203e-4 ± 2.73215369e-5 | 1.38782810e-2 ± 3.68367833e-4 | 17.06 ± 0.7306 | reported |
| ising_open_64x64 | n=4096, open 64x64 | 3.28229284e-3 ± 9.84237700e-5 | 2.93838190e-2 ± 2.34235872e-3 | 8.952 ± 0.7625 | reported |
| potts_open_q3_4x4 | n=16, q=3, open Potts 4x4 | 3.25698304e-5 ± 9.86201163e-7 | 7.11206083e-3 ± 1.28962043e-4 | 218.4 ± 7.707 | reported |
| potts_open_q3_8x8 | n=64, q=3, open Potts 8x8 | 1.27207981e-4 ± 2.89124543e-6 | 8.06332649e-3 ± 2.89496559e-4 | 63.39 ± 2.693 | reported |
| potts_open_q4_4x4 | n=16, q=4, open Potts 4x4 | 3.84741626e-5 ± 1.16441059e-6 | 9.28341410e-3 ± 6.75426200e-4 | 241.3 ± 19.01 | reported |
| potts_open_q4_8x8 | n=64, q=4, open Potts 8x8 | 1.63414243e-4 ± 4.20761326e-6 | 9.64319856e-3 ± 5.80200866e-4 | 59.01 ± 3.862 | reported |
| tsp_iv4_three_body_d12 | n=12, three-body d12 | 2.16495705e-4 ± 6.05636804e-6 | 2.29838563e-1 ± 3.71005722e-2 | 1062 ± 173.9 | reported |
| tsp_iv4_three_body_d16 | n=16, three-body d16 | 5.22169163e-4 ± 1.45167747e-5 | 5.93470651e-2 ± 1.34364699e-2 | 113.7 ± 25.93 | reported |

Student-t, quantization, cadence/edge, and bracket for each core summary are the receipt fields. Their RSS matches `uncertainty_combined` in `attempt-037/report.json`.

## Intel RAPL package

Attribution refused on 300 of 300 trials (bracket drift). No J/ESS. Gross and incremental joules are the receipt summary means.

| workload | side | n | gross J mean | incremental J mean | J/ESS |
| --- | --- | --- | --- | --- | --- |
| thrml_readme_ising_chain_n5 | ferrotherm | 10 | 1.68421794e-2 | 1.35732276e-2 | refused |
| thrml_readme_ising_chain_n5 | thrml | 10 | 1.44270891e1 | 1.22052528e1 | refused |
| ising_grid2d_open_3x3 | ferrotherm | 10 | 2.02417124e-2 | 1.64352027e-2 | refused |
| ising_grid2d_open_3x3 | thrml | 10 | 1.06911591e1 | 9.01775703e0 | refused |
| potts_path_q3_n6 | ferrotherm | 10 | 2.10016000e-2 | 1.72062922e-2 | refused |
| potts_path_q3_n6 | thrml | 10 | 3.20677554e1 | 2.63582800e1 | refused |
| tsp_iv4_three_body_d8 | ferrotherm | 10 | 1.85583240e-2 | 1.49292738e-2 | refused |
| tsp_iv4_three_body_d8 | thrml | 10 | 2.35344531e1 | 1.95785023e1 | refused |
| ising_open_4x4 | ferrotherm | 10 | 1.87374874e-1 | 1.51631216e-1 | refused |
| ising_open_4x4 | thrml | 10 | 1.04162927e1 | 8.66082980e0 | refused |
| ising_open_8x8 | ferrotherm | 10 | 6.47509127e-2 | 5.25290830e-2 | refused |
| ising_open_8x8 | thrml | 10 | 1.08613436e1 | 9.02199607e0 | refused |
| ising_open_16x16 | ferrotherm | 10 | 2.21165582e-1 | 1.77917667e-1 | refused |
| ising_open_16x16 | thrml | 10 | 1.12374065e1 | 9.36422329e0 | refused |
| ising_open_32x32 | ferrotherm | 10 | 8.61984067e-1 | 6.92905295e-1 | refused |
| ising_open_32x32 | thrml | 10 | 1.53173665e1 | 1.27197641e1 | refused |
| ising_open_64x64 | ferrotherm | 10 | 3.55586596e0 | 2.87942866e0 | refused |
| ising_open_64x64 | thrml | 10 | 3.16972608e1 | 2.67585866e1 | refused |
| potts_open_q3_4x4 | ferrotherm | 10 | 5.93693697e-2 | 4.80431455e-2 | refused |
| potts_open_q3_4x4 | thrml | 10 | 1.46553015e1 | 1.20189602e1 | refused |
| potts_open_q3_8x8 | ferrotherm | 10 | 2.22735746e-1 | 1.80003038e-1 | refused |
| potts_open_q3_8x8 | thrml | 10 | 1.58052772e1 | 1.30283584e1 | refused |
| potts_open_q4_4x4 | ferrotherm | 10 | 7.46666257e-2 | 6.04196430e-2 | refused |
| potts_open_q4_4x4 | thrml | 10 | 1.85110447e1 | 1.51800650e1 | refused |
| potts_open_q4_8x8 | ferrotherm | 10 | 3.08540706e-1 | 2.51054442e-1 | refused |
| potts_open_q4_8x8 | thrml | 10 | 1.90675762e1 | 1.56772352e1 | refused |
| tsp_iv4_three_body_d12 | ferrotherm | 10 | 3.94796831e-2 | 3.18476859e-2 | refused |
| tsp_iv4_three_body_d12 | thrml | 10 | 3.17129467e1 | 2.64062515e1 | refused |
| tsp_iv4_three_body_d16 | ferrotherm | 10 | 3.03567574e-1 | 2.43598277e-1 | refused |
| tsp_iv4_three_body_d16 | thrml | 10 | 3.44812869e1 | 2.88705393e1 | refused |

## NVML RTX 4050

Attribution refused on 300 of 300 trials. The drift stored on these rows is the package idle bracket, not a separate GPU baseline. No J/ESS.

| workload | side | n | gross J mean | incremental J mean | J/ESS |
| --- | --- | --- | --- | --- | --- |
| thrml_readme_ising_chain_n5 | ferrotherm | 10 | 1.99677368e-1 | 1.96452052e-1 | refused |
| thrml_readme_ising_chain_n5 | thrml | 10 | 1.04132619e2 | 1.02594615e2 | refused |
| ising_grid2d_open_3x3 | ferrotherm | 10 | 2.37463819e-1 | 2.33707327e-1 | refused |
| ising_grid2d_open_3x3 | thrml | 10 | 1.23059230e2 | 1.21324795e2 | refused |
| potts_path_q3_n6 | ferrotherm | 10 | 2.36133452e-1 | 2.32365625e-1 | refused |
| potts_path_q3_n6 | thrml | 10 | 4.35891150e2 | 4.30098662e2 | refused |
| tsp_iv4_three_body_d8 | ferrotherm | 10 | 2.19583805e-1 | 2.16022624e-1 | refused |
| tsp_iv4_three_body_d8 | thrml | 10 | 3.05307000e2 | 3.01168576e2 | refused |
| ising_open_4x4 | ferrotherm | 10 | 2.19612966e0 | 2.16087388e0 | refused |
| ising_open_4x4 | thrml | 10 | 1.26050327e2 | 1.24262215e2 | refused |
| ising_open_8x8 | ferrotherm | 10 | 7.44763964e-1 | 7.32642431e-1 | refused |
| ising_open_8x8 | thrml | 10 | 1.24966140e2 | 1.23028401e2 | refused |
| ising_open_16x16 | ferrotherm | 10 | 2.65480440e0 | 2.61232571e0 | refused |
| ising_open_16x16 | thrml | 10 | 1.26945800e2 | 1.25016143e2 | refused |
| ising_open_32x32 | ferrotherm | 10 | 1.05073464e1 | 1.03416477e1 | refused |
| ising_open_32x32 | thrml | 10 | 1.93005925e2 | 1.90299472e2 | refused |
| ising_open_64x64 | ferrotherm | 10 | 4.19667000e1 | 4.12991198e1 | refused |
| ising_open_64x64 | thrml | 10 | 3.53329300e2 | 3.48182028e2 | refused |
| potts_open_q3_4x4 | ferrotherm | 10 | 7.05103372e-1 | 6.93855355e-1 | refused |
| potts_open_q3_4x4 | thrml | 10 | 1.81864550e2 | 1.79222252e2 | refused |
| potts_open_q3_8x8 | ferrotherm | 10 | 2.62597745e0 | 2.58367790e0 | refused |
| potts_open_q3_8x8 | thrml | 10 | 2.05826400e2 | 2.02999082e2 | refused |
| potts_open_q4_4x4 | ferrotherm | 10 | 8.65759960e-1 | 8.51599039e-1 | refused |
| potts_open_q4_4x4 | thrml | 10 | 2.14272267e2 | 2.11004723e2 | refused |
| potts_open_q4_8x8 | ferrotherm | 10 | 3.60504127e0 | 3.54804280e0 | refused |
| potts_open_q4_8x8 | thrml | 10 | 2.42240667e2 | 2.38790454e2 | refused |
| tsp_iv4_three_body_d12 | ferrotherm | 10 | 4.68057367e-1 | 4.60589891e-1 | refused |
| tsp_iv4_three_body_d12 | thrml | 10 | 3.66537050e2 | 3.61259015e2 | refused |
| tsp_iv4_three_body_d16 | ferrotherm | 10 | 3.74818092e0 | 3.68856924e0 | refused |
| tsp_iv4_three_body_d16 | thrml | 10 | 4.11673100e2 | 4.06026412e2 | refused |

## `ising_open_4x4` is under investigation

Intel RAPL core-domain J/ESS on jetson-hub vs THRML 0.1.4 on JAX CPU, per workload. Tarball sha256 1df27d9f6716ca6ad0da9a0fe4c773dfc8f5a46609e5b938cc086a00d7f1b613. Publish commit 22d3e720714c76e73389998d36e5b8159e89fe96.

The ferrotherm core mean on that row is `1.22954420e-4`, above the open 8×8 mean `5.19702797e-5`. The quotient is not cited. The traces say the difference is a real per-repetition cost.

Accuracy-phase ferrotherm means (10 trials; the sampler timer stops before scoring):

| lattice | tau_int | ESS | sampler seconds |
| --- | --- | --- | --- |
| open 3×3 (n=9) | 0.899 | 1127 | 6.29e-4 |
| open 4×4 (n=16) | 0.780 | 1306 | 8.84e-4 |
| open 8×8 (n=64) | 1.087 | 963 | 2.59e-3 |
| open 16×16 (n=256) | 1.097 | 923 | 9.77e-3 |

Energy-window ferrotherm means, Intel RAPL core (10 trials; the counter wraps the whole repetition, including the score):

| lattice | window s | repetitions | incremental J | ESS | seconds / repetition |
| --- | --- | --- | --- | --- | --- |
| open 3×3 | 2.0005 | 2327.1 | 1.692e-2 | 1123 | 8.60e-4 |
| open 4×4 | 2.0041 | 253.1 | 1.566e-1 | 1273 | 7.92e-3 |
| open 8×8 | 2.0009 | 748.4 | 5.431e-2 | 1045 | 2.67e-3 |
| open 16×16 | 2.0066 | 209.8 | 1.844e-1 | 929 | 9.56e-3 |

ESS is higher on the 4×4 than on the 8×8, and `tau_int` is lower. The trial-0 energy trace (2000 draws) has an initial-positive-sequence autocorrelation near 0.75. Autocorrelation is not the cause.

`ising_open_scaled` leaves `exact_energy` unset when `side <= 4`, and the oracle string on this receipt is "transfer matrix cross-checked by enumeration". `score_ising` then calls `exact_boltzmann`, which enumerates the 2^16 law, on every repetition. Side 8 stores the transfer-matrix mean and scores with `score_mean_only`. The energy window calls `repeat_until_window` → `plan_ess` → `ft::run`, and `run_ising` scores after the sampler timer returns, still inside that window. The 4×4 accuracy timer records 0.88 ms; the metered repetition takes 7.92 ms and 0.157 core-joules. The 8×8 accuracy timer (2.59 ms) and the metered repetition (2.67 ms) agree. The open 3×3 also leaves `exact_energy` unset, and enumerating 2^9 leaves the accuracy time (0.63 ms) and the metered time (0.86 ms) in the same range. The 4×4 core J/ESS is the cost of enumerating 2^16 inside the 2 s window. It is not a sampling result.

