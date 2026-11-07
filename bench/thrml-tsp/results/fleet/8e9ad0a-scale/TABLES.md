# Framework CPU scale suite — 8e9ad0a4

Exact revision: 8e9ad0a4f1e13b25f3abfcab0a37576fb55c9824. THRML backend: thrml=0.1.4 jax=0.12.0.1 jaxlib=0.12.0.1 equinox=0.13.8 device=cpu:0 jax_platform=cpu. Framework only; no Mac or thor.local access.

Primary instrument RAPL, package scope. Grade: **contended host**. Quiet first returned load_gated. Each workload attempted five trials of each sampler; all 15 workloads and 150 trials finished, no watchdog timeout.

Contended baseline: 144.811812 W; sigma 3.47039332 W. Trial load1 range 11.49–23.85. Only 101/150 were marked attributable by the harness.

## Critical interpretation

143/150 incremental-energy trials are negative; 99 negative trials nevertheless have attributable=true. All available per-workload J/ESS means are negative. These are recorded signed baseline-subtracted measurements, NOT usable positive energy costs. No energy-efficiency gain ratio is supported by this run. Arithmetic quotients of two negative means are supplied only for transparency and are not speedup/efficiency ratios. Nulls and negative values were preserved. No rerun, clamping, baseline replacement or selective exclusion was used.

The report’s generic oracle field says exact enumeration even for mean-only workloads. Scope below follows workload source metadata: the 32×32 and 64×64 Ising reference is an infinite-lattice limit, not an exact finite-grid oracle. Mean-energy agreement does not establish full-distribution agreement.

## J/ESS measurements

Cells copy the harness’s mean [95% CI] and metric-specific n without recomputing. These are incremental J per effective sample. Attribution counts are the unmodified harness flags, not independent validation.

|Workload|Sampler|J/ESS [95% CI]; metric n|Attributable/attempted|load1 range|
|---|---|---|---|---|
|thrml_readme_ising_chain_n5|ferrotherm|-1.26830603e-05 [-1.78698153e-05, -7.49630524e-06]; n=4|4/5|21.47–23.85|
|thrml_readme_ising_chain_n5|thrml|-6.93292818e-03 [-9.44538632e-03, -4.42047005e-03]; n=4|4/5|21.47–23.76|
|ising_grid2d_open_3x3|ferrotherm|-2.35521534e-05 [-3.19769340e-05, -1.51273729e-05]; n=5|5/5|21.23–23.76|
|ising_grid2d_open_3x3|thrml|-9.18922935e-03 [-1.32420923e-02, -5.13636640e-03]; n=4|4/5|21.23–23.46|
|potts_path_q3_n6|ferrotherm|-9.67984036e-06 [-1.82125258e-05, -1.14715489e-06]; n=3|3/5|20.57–21.33|
|potts_path_q3_n6|thrml|-1.93260890e-02 [-2.80631466e-02, -1.05890314e-02]; n=4|4/5|20.57–21.33|
|tsp_iv4_three_body_d8|ferrotherm|-5.94505599e-05 [-7.32952937e-05, -4.56058261e-05]; n=5|5/5|18.85–20.64|
|tsp_iv4_three_body_d8|thrml|-8.31072134e-02 [-1.22665491e-01, -4.35489356e-02]; n=5|5/5|18.85–20.64|
|ising_open_4x4|ferrotherm|-1.97172899e-04 [-2.68021342e-04, -1.26324455e-04]; n=5|5/5|17.34–19.5|
|ising_open_4x4|thrml|-8.80062590e-03 [-2.01282037e-02, 2.52695185e-03]; n=5|5/5|17.34–19.5|
|ising_open_8x8|ferrotherm|-2.10459932e-05 [-2.00648881e-04, 1.58556895e-04]; n=3|3/5|18.04–19.42|
|ising_open_8x8|thrml|-8.30437743e-03 [-1.06599400e-02, -5.94881490e-03]; n=3|3/5|18.04–19.42|
|ising_open_16x16|ferrotherm|-2.96541735e-04 [-4.71274236e-04, -1.21809233e-04]; n=3|3/5|17.21–18.25|
|ising_open_16x16|thrml|-1.13002180e-02 [-1.56266765e-02, -6.97375943e-03]; n=4|4/5|17.21–18.25|
|ising_open_32x32|ferrotherm|-1.31824875e-03 [-2.02231043e-03, -6.14187064e-04]; n=4|4/5|17.91–19.19|
|ising_open_32x32|thrml|-2.38670641e-02 [-5.02458990e-02, 2.51177080e-03]; n=3|3/5|17.91–19.19|
|ising_open_64x64|ferrotherm|-1.01928380e-02 [-1.12587243e-02, -9.12695179e-03]; n=5|5/5|15.99–18.54|
|ising_open_64x64|thrml|-4.07114723e-02 [-4.93322313e-02, -3.20907133e-02]; n=5|5/5|15.99–17.77|
|potts_open_q3_4x4|ferrotherm|-6.32104760e-05 [-7.95377822e-05, -4.68831697e-05]; n=5|5/5|14.62–15.75|
|potts_open_q3_4x4|thrml|-1.52098098e-02 [-1.80973164e-02, -1.23223032e-02]; n=5|5/5|14.62–15.75|
|potts_open_q3_8x8|ferrotherm|-3.12220247e-04 [-3.31784470e-04, -2.92656024e-04]; n=5|5/5|12.87–14.17|
|potts_open_q3_8x8|thrml|-2.00829253e-02 [-2.35468499e-02, -1.66190006e-02]; n=5|5/5|12.48–14.17|
|potts_open_q4_4x4|ferrotherm|-8.94229338e-05 [-9.06337827e-05, -8.82120848e-05]; n=4|4/5|11.49–12.48|
|potts_open_q4_4x4|thrml|-2.13453656e-02 [-2.30960932e-02, -1.95946381e-02]; n=3|3/5|11.49–12.12|
|potts_open_q4_8x8|ferrotherm|unavailable|0/5|12.09–14.29|
|potts_open_q4_8x8|thrml|unavailable|0/5|12.64–14.67|
|tsp_iv4_three_body_d12|ferrotherm|unavailable|0/5|14.67–16.23|
|tsp_iv4_three_body_d12|thrml|unavailable|0/5|15.02–16.23|
|tsp_iv4_three_body_d16|ferrotherm|unavailable|0/5|16.53–17.67|
|tsp_iv4_three_body_d16|thrml|unavailable|0/5|16.53–17.86|

## Ratios

Raw quotient = THRML mean signed J/ESS divided by ferrotherm mean signed J/ESS. This is a derived arithmetic quotient, not a measured or valid efficiency-gain ratio. No ratio CI is asserted.

|Workload|Raw signed quotient T/F|Valid efficiency ratio|
|---|---|---|
|thrml_readme_ising_chain_n5|5.46628969e+02|unsupported: negative prices|
|ising_grid2d_open_3x3|3.90165145e+02|unsupported: negative prices|
|potts_path_q3_n6|1.99652972e+03|unsupported: negative prices|
|tsp_iv4_three_body_d8|1.39792146e+03|unsupported: negative prices|
|ising_open_4x4|4.46340544e+01|unsupported: negative prices|
|ising_open_8x8|3.94582349e+02|unsupported: negative prices|
|ising_open_16x16|3.81066699e+01|unsupported: negative prices|
|ising_open_32x32|1.81051293e+01|unsupported: negative prices|
|ising_open_64x64|3.99412532e+00|unsupported: negative prices|
|potts_open_q3_4x4|2.40621662e+02|unsupported: negative prices|
|potts_open_q3_8x8|6.43229435e+01|unsupported: negative prices|
|potts_open_q4_4x4|2.38701245e+02|unsupported: negative prices|
|potts_open_q4_8x8|unavailable|unavailable|
|tsp_iv4_three_body_d12|unavailable|unavailable|
|tsp_iv4_three_body_d16|unavailable|unavailable|

## Oracle-reference accuracy

Absolute-error cells are unmodified harness means [95% CI] across five accuracy chains. The reference-coverage column is a descriptive derived check: does the declared reference lie inside a two-sided Student-t 95% CI of the five chain mean energies? t4=2.7764451052; computed from serialized trial values. This is not a preregistered PASS threshold or simultaneous/multiple-testing-adjusted certification.

|Workload|Reference scope|Reference energy|F absolute mean-energy error [95% CI]|T absolute mean-energy error [95% CI]|Reference in mean-energy CI F/T|
|---|---|---|---|---|---|
|thrml_readme_ising_chain_n5|finite Boltzmann enumeration|-9.24234315e-01|9.05313710e-03 [2.93582155e-03, 1.51704526e-02]; n=5|2.69468629e-02 [9.76258979e-03, 4.41311360e-02]; n=5|inside/inside|
|ising_grid2d_open_3x3|finite Boltzmann enumeration|-2.72044887e+00|2.92693200e-02 [6.31725745e-03, 5.22213825e-02]; n=5|3.24897733e-02 [-7.26723775e-04, 6.57062704e-02]; n=5|inside/inside|
|potts_path_q3_n6|finite Boltzmann enumeration|-8.54467121e-01|1.30934241e-02 [1.16665170e-03, 2.50201966e-02]; n=5|6.29342413e-03 [-8.07333141e-04, 1.33941814e-02]; n=5|inside/inside|
|tsp_iv4_three_body_d8|finite Boltzmann enumeration|-9.83314127e+00|1.03089746e-01 [5.30574346e-02, 1.53122057e-01]; n=5|6.24463772e-02 [1.33935096e-02, 1.11499245e-01]; n=5|inside/inside|
|ising_open_4x4|finite enumeration + transfer matrix|-7.95222265e+00|1.25044531e-01 [-2.88033052e-02, 2.78892367e-01]; n=5|1.40155469e-01 [4.34663758e-02, 2.36844562e-01]; n=5|inside/inside|
|ising_open_8x8|finite open-grid transfer matrix; Kac-Ward cross-check|-3.82506796e+01|4.00264076e-01 [4.85969631e-02, 7.51931190e-01]; n=5|4.06392229e-01 [1.82284188e-01, 6.30500271e-01]; n=5|inside/inside|
|ising_open_16x16|finite open-grid transfer matrix|-1.66480115e+02|5.29268718e-01 [1.25846498e-01, 9.32690939e-01]; n=5|9.44731282e-01 [2.41488664e-01, 1.64797390e+00]; n=5|inside/inside|
|ising_open_32x32|Onsager infinite-lattice energy × N; NOT finite-grid oracle|-7.21407049e+02|2.88374485e+01 [2.62080025e+01, 3.14668946e+01]; n=5|2.89600485e+01 [2.77024783e+01, 3.02176188e+01]; n=5|outside/outside|
|ising_open_64x64|Onsager infinite-lattice energy × N; NOT finite-grid oracle|-2.88562819e+03|5.25345941e+01 [4.72439218e+01, 5.78252665e+01]; n=5|5.65071941e+01 [5.21097909e+01, 6.09045974e+01]; n=5|outside/outside|
|potts_open_q3_4x4|finite Potts transfer matrix|-1.10065195e+01|3.26195016e-02 [1.79470831e-03, 6.34442949e-02]; n=5|6.21960997e-02 [1.52157704e-03, 1.22870622e-01]; n=5|outside/inside|
|potts_open_q3_8x8|finite Potts transfer matrix|-5.15118060e+01|5.79163706e-02 [-2.10812458e-02, 1.36913987e-01]; n=5|8.25612098e-02 [-1.17956082e-02, 1.76918028e-01]; n=5|inside/inside|
|potts_open_q4_4x4|finite Potts transfer matrix|-8.60597948e+00|3.77041031e-02 [-2.39675362e-02, 9.93757425e-02]; n=5|6.31958969e-02 [2.21203871e-02, 1.04271407e-01]; n=5|inside/inside|
|potts_open_q4_8x8|finite Potts transfer matrix|-4.02426104e+01|9.90220731e-02 [-6.19477906e-03, 2.04238925e-01]; n=5|8.28337807e-02 [3.52277571e-02, 1.30439804e-01]; n=5|inside/inside|
|tsp_iv4_three_body_d12|finite Boltzmann enumeration|-1.35236766e+01|1.40759569e-01 [6.75273259e-02, 2.13991813e-01]; n=5|1.42770276e-01 [-4.67038088e-03, 2.90210933e-01]; n=5|inside/inside|
|tsp_iv4_three_body_d16|finite Boltzmann enumeration|-2.19360838e+01|1.21224261e-01 [4.84543859e-02, 1.93994137e-01]; n=5|9.44983616e-02 [1.05818963e-02, 1.78414827e-01]; n=5|inside/inside|

### Derived mean-energy intervals

|Workload|F mean energy [95% CI]|T mean energy [95% CI]|
|---|---|---|
|thrml_readme_ising_chain_n5|-9.27900000e-01 [-9.40918242e-01, -9.14881758e-01]; n=5|-9.17900000e-01 [-9.58116307e-01, -8.77683693e-01]; n=5|
|ising_grid2d_open_3x3|-2.71360000e+00 [-2.75928792e+00, -2.66791208e+00]; n=5|-2.71220000e+00 [-2.76703145e+00, -2.65736855e+00]; n=5|
|potts_path_q3_n6|-8.54360000e-01 [-8.76099689e-01, -8.32620311e-01]; n=5|-8.50840000e-01 [-8.60909581e-01, -8.40770419e-01]; n=5|
|tsp_iv4_three_body_d8|-9.77553178e+00 [-9.90432685e+00, -9.64673671e+00]; n=5|-9.82887970e+00 [-9.92830932e+00, -9.72945009e+00]; n=5|
|ising_open_4x4|-7.97500000e+00 [-8.20478837e+00, -7.74521163e+00]; n=5|-8.04100000e+00 [-8.21993106e+00, -7.86206894e+00]; n=5|
|ising_open_8x8|-3.82076000e+01 [-3.88624636e+01, -3.75527364e+01]; n=5|-3.85208000e+01 [-3.89981769e+01, -3.80434231e+01]; n=5|
|ising_open_16x16|-1.66184400e+02 [-1.66915202e+02, -1.65453598e+02]; n=5|-1.66741200e+02 [-1.68184533e+02, -1.65297867e+02]; n=5|
|ising_open_32x32|-6.92569600e+02 [-6.95199046e+02, -6.89940154e+02]; n=5|-6.92447000e+02 [-6.93704570e+02, -6.91189430e+02]; n=5|
|ising_open_64x64|-2.83309360e+03 [-2.83838427e+03, -2.82780293e+03]; n=5|-2.82912100e+03 [-2.83351840e+03, -2.82472360e+03]; n=5|
|potts_open_q3_4x4|-1.09739000e+01 [-1.10047248e+01, -1.09430752e+01]; n=5|-1.10341000e+01 [-1.11324381e+01, -1.09357619e+01]; n=5|
|potts_open_q3_8x8|-5.15654000e+01 [-5.16500729e+01, -5.14807271e+01]; n=5|-5.14962000e+01 [-5.16430675e+01, -5.13493325e+01]; n=5|
|potts_open_q4_4x4|-8.63330000e+00 [-8.70474641e+00, -8.56185359e+00]; n=5|-8.58660000e+00 [-8.67965897e+00, -8.49354103e+00]; n=5|
|potts_open_q4_8x8|-4.02463000e+01 [-4.04193344e+01, -4.00732656e+01]; n=5|-4.02884000e+01 [-4.03953989e+01, -4.01814011e+01]; n=5|
|tsp_iv4_three_body_d12|-1.34857871e+01 [-1.36877268e+01, -1.32838474e+01]; n=5|-1.36664027e+01 [-1.38139257e+01, -1.35188798e+01]; n=5|
|tsp_iv4_three_body_d16|-2.19205201e+01 [-2.21025885e+01, -2.17384517e+01]; n=5|-2.19682299e+01 [-2.21174275e+01, -2.18190322e+01]; n=5|

## Anomalies and limits

- 49 trials were not attributable because incremental energy was inside the idle band. Their J/ESS values remain null.
- Both samplers have 0/5 attributable trials for q=4 Potts 8×8, three-body d=12 and three-body d=16.
- 99 negative-incremental trials carry attributable=true. All nonnull workload mean J/ESS estimates are negative. No positive price/gain ratio is supported.
- Both samplers’ mean-energy CIs exclude the 32×32 and 64×64 infinite-lattice references. Finite-grid boundary mismatch is not a proof of sampler failure.
- For finite-reference workloads, ferrotherm q=3 Potts 4×4 reference is just outside its five-chain mean-energy CI. All other finite-reference mean-energy CIs include the reference for both samplers; these are descriptive unadjusted checks.
- 4×4 Ising distribution TV is about 0.8 for both samplers with 2,000 draws over 65,536 possible states. Mean agreement is not a full-distribution certification. Larger mean-only workloads have null TV/KL, not zeros.
- Evidence grade remains contended host; CI scatter does not cover all systematic baseline drift or background-load uncertainty.
- No cell timeout fired. Watchdog bound: 180 s without log progress, 1,800 s whole invocation; unsupported per-cell isolation is disclosed in driver.py.
- AwaitShell could not read the framework archive job’s terminal path (outside allowed root); archive completion was subsequently confirmed from the ordinary result file and successful extraction. No refused path was accessed.


- Provenance git_dirty=true is the generated untracked Python bytecode cache. Tracked diff against the pinned commit is empty (exit 0); this was verified and recorded, not silently called clean.
