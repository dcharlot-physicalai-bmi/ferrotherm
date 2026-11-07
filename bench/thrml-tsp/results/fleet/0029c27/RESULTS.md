# 0029c27

Revision `0029c270ea401109254f691ad78bc1666c9540df`. Both hosts are evidence grade **contended host**. 40 of 40 measured trials on each host were attributable. J/ESS is incremental joules per effective sample, per sampler repetition: mean and the harness 95% Student-t interval, n=5. The cells are copied from `TABLES.md`. They are trial scatter, not a budget for baseline drift or background load.

RAPL package and macmon whole-system are separate measurements. The tables below are not one comparison.

## RAPL package, `framework-128gb-0000` (primary)

| Workload | ferrotherm J/ESS | THRML J/ESS |
|---|---|---|
| thrml_readme_ising_chain_n5 | 4.73864417e-06 [4.45701112e-06, 5.02027723e-06] | 2.77753027e-03 [2.40850973e-03, 3.14655081e-03] |
| ising_grid2d_open_3x3 | 8.47697947e-06 [7.58536698e-06, 9.36859196e-06] | 4.48414457e-03 [4.42244895e-03, 4.54584019e-03] |
| potts_path_q3_n6 | 5.09328891e-06 [4.33088027e-06, 5.85569755e-06] | 7.65889751e-03 [7.07170181e-03, 8.24609322e-03] |
| tsp_iv4_three_body_d8 | 1.74431866e-05 [1.67304690e-05, 1.81559042e-05] | 2.97110511e-02 [2.02037979e-02, 3.92183044e-02] |

## macmon whole-system, M5 Max 48 GB (secondary)

| Workload | ferrotherm J/ESS | THRML J/ESS |
|---|---|---|
| thrml_readme_ising_chain_n5 | 4.40969033e-06 [4.25611040e-06, 4.56327027e-06] | 1.56413579e-03 [1.32624936e-03, 1.80202222e-03] |
| ising_grid2d_open_3x3 | 8.06818676e-06 [6.86359572e-06, 9.27277780e-06] | 2.80065807e-03 [2.47721379e-03, 3.12410234e-03] |
| potts_path_q3_n6 | 4.91204208e-06 [4.48560206e-06, 5.33848211e-06] | 5.79050226e-03 [5.44504263e-03, 6.13596188e-03] |
| tsp_iv4_three_body_d8 | 1.89290013e-05 [1.64946008e-05, 2.13634017e-05] | 2.30589743e-02 [1.75634105e-02, 2.85545381e-02] |

## Limits

- Contended host. The Framework quiet attempt was `load_gated`. The M5 run was `--contended` and stays contended host. These figures are not quiet-host evidence.
- THRML 0.1.4 ran on JAX 0.12.0.1 CPU. There is no GPU THRML result here.
- The models are small: a 5-spin chain, a 3×3 Ising grid, a 6-site Potts path, and an 8-spin three-body system. Each cell is five trials.
- No quiet-host leadership claim. This bundle does not say which sampler costs less on a quiet machine, and it does not speak for a Z1 device.
