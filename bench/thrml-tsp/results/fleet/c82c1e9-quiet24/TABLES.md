# Tables

Intel RAPL core-domain J/ESS on jetson-hub vs THRML 0.1.4 on JAX CPU, per workload. Tarball sha256 1df27d9f6716ca6ad0da9a0fe4c773dfc8f5a46609e5b938cc086a00d7f1b613. Publish commit 22d3e720714c76e73389998d36e5b8159e89fe96.

Small accuracy-suite workloads through the open 64×64 (n=4096). Not a large-grid energy claim. Meters stay separate. The open 64×64 core-domain quotient is 8.952 ± 0.762, about 9×, on that lattice only. `ising_open_4x4` is under investigation and is not a cited quotient.

## Intel RAPL core J/ESS

| workload | size | ferrotherm core J/ESS ± combined | THRML core J/ESS ± combined | THRML / ferrotherm | status |
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

## Intel RAPL package, joules only

300 of 300 trials refused attribution. No J/ESS.

| workload | side | gross J mean | incremental J mean |
| --- | --- | --- | --- |
| thrml_readme_ising_chain_n5 | ferrotherm | 1.68421794e-2 | 1.35732276e-2 |
| thrml_readme_ising_chain_n5 | thrml | 1.44270891e1 | 1.22052528e1 |
| ising_grid2d_open_3x3 | ferrotherm | 2.02417124e-2 | 1.64352027e-2 |
| ising_grid2d_open_3x3 | thrml | 1.06911591e1 | 9.01775703e0 |
| potts_path_q3_n6 | ferrotherm | 2.10016000e-2 | 1.72062922e-2 |
| potts_path_q3_n6 | thrml | 3.20677554e1 | 2.63582800e1 |
| tsp_iv4_three_body_d8 | ferrotherm | 1.85583240e-2 | 1.49292738e-2 |
| tsp_iv4_three_body_d8 | thrml | 2.35344531e1 | 1.95785023e1 |
| ising_open_4x4 | ferrotherm | 1.87374874e-1 | 1.51631216e-1 |
| ising_open_4x4 | thrml | 1.04162927e1 | 8.66082980e0 |
| ising_open_8x8 | ferrotherm | 6.47509127e-2 | 5.25290830e-2 |
| ising_open_8x8 | thrml | 1.08613436e1 | 9.02199607e0 |
| ising_open_16x16 | ferrotherm | 2.21165582e-1 | 1.77917667e-1 |
| ising_open_16x16 | thrml | 1.12374065e1 | 9.36422329e0 |
| ising_open_32x32 | ferrotherm | 8.61984067e-1 | 6.92905295e-1 |
| ising_open_32x32 | thrml | 1.53173665e1 | 1.27197641e1 |
| ising_open_64x64 | ferrotherm | 3.55586596e0 | 2.87942866e0 |
| ising_open_64x64 | thrml | 3.16972608e1 | 2.67585866e1 |
| potts_open_q3_4x4 | ferrotherm | 5.93693697e-2 | 4.80431455e-2 |
| potts_open_q3_4x4 | thrml | 1.46553015e1 | 1.20189602e1 |
| potts_open_q3_8x8 | ferrotherm | 2.22735746e-1 | 1.80003038e-1 |
| potts_open_q3_8x8 | thrml | 1.58052772e1 | 1.30283584e1 |
| potts_open_q4_4x4 | ferrotherm | 7.46666257e-2 | 6.04196430e-2 |
| potts_open_q4_4x4 | thrml | 1.85110447e1 | 1.51800650e1 |
| potts_open_q4_8x8 | ferrotherm | 3.08540706e-1 | 2.51054442e-1 |
| potts_open_q4_8x8 | thrml | 1.90675762e1 | 1.56772352e1 |
| tsp_iv4_three_body_d12 | ferrotherm | 3.94796831e-2 | 3.18476859e-2 |
| tsp_iv4_three_body_d12 | thrml | 3.17129467e1 | 2.64062515e1 |
| tsp_iv4_three_body_d16 | ferrotherm | 3.03567574e-1 | 2.43598277e-1 |
| tsp_iv4_three_body_d16 | thrml | 3.44812869e1 | 2.88705393e1 |

## NVML RTX 4050, joules only

300 of 300 trials refused attribution. The idle bracket on these rows is the package bracket. No J/ESS.

| workload | side | gross J mean | incremental J mean |
| --- | --- | --- | --- |
| thrml_readme_ising_chain_n5 | ferrotherm | 1.99677368e-1 | 1.96452052e-1 |
| thrml_readme_ising_chain_n5 | thrml | 1.04132619e2 | 1.02594615e2 |
| ising_grid2d_open_3x3 | ferrotherm | 2.37463819e-1 | 2.33707327e-1 |
| ising_grid2d_open_3x3 | thrml | 1.23059230e2 | 1.21324795e2 |
| potts_path_q3_n6 | ferrotherm | 2.36133452e-1 | 2.32365625e-1 |
| potts_path_q3_n6 | thrml | 4.35891150e2 | 4.30098662e2 |
| tsp_iv4_three_body_d8 | ferrotherm | 2.19583805e-1 | 2.16022624e-1 |
| tsp_iv4_three_body_d8 | thrml | 3.05307000e2 | 3.01168576e2 |
| ising_open_4x4 | ferrotherm | 2.19612966e0 | 2.16087388e0 |
| ising_open_4x4 | thrml | 1.26050327e2 | 1.24262215e2 |
| ising_open_8x8 | ferrotherm | 7.44763964e-1 | 7.32642431e-1 |
| ising_open_8x8 | thrml | 1.24966140e2 | 1.23028401e2 |
| ising_open_16x16 | ferrotherm | 2.65480440e0 | 2.61232571e0 |
| ising_open_16x16 | thrml | 1.26945800e2 | 1.25016143e2 |
| ising_open_32x32 | ferrotherm | 1.05073464e1 | 1.03416477e1 |
| ising_open_32x32 | thrml | 1.93005925e2 | 1.90299472e2 |
| ising_open_64x64 | ferrotherm | 4.19667000e1 | 4.12991198e1 |
| ising_open_64x64 | thrml | 3.53329300e2 | 3.48182028e2 |
| potts_open_q3_4x4 | ferrotherm | 7.05103372e-1 | 6.93855355e-1 |
| potts_open_q3_4x4 | thrml | 1.81864550e2 | 1.79222252e2 |
| potts_open_q3_8x8 | ferrotherm | 2.62597745e0 | 2.58367790e0 |
| potts_open_q3_8x8 | thrml | 2.05826400e2 | 2.02999082e2 |
| potts_open_q4_4x4 | ferrotherm | 8.65759960e-1 | 8.51599039e-1 |
| potts_open_q4_4x4 | thrml | 2.14272267e2 | 2.11004723e2 |
| potts_open_q4_8x8 | ferrotherm | 3.60504127e0 | 3.54804280e0 |
| potts_open_q4_8x8 | thrml | 2.42240667e2 | 2.38790454e2 |
| tsp_iv4_three_body_d12 | ferrotherm | 4.68057367e-1 | 4.60589891e-1 |
| tsp_iv4_three_body_d12 | thrml | 3.66537050e2 | 3.61259015e2 |
| tsp_iv4_three_body_d16 | ferrotherm | 3.74818092e0 | 3.68856924e0 |
| tsp_iv4_three_body_d16 | thrml | 4.11673100e2 | 4.06026412e2 |

