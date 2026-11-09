# Post-hoc sensitivity, not a grade

This is a selection on the numbers `attempt-003/report.json` already stored.
It is not a re-grade under `c82c1e96`, and it does not change any joule.

Harness `5c54005` wrote one busy fraction per core outside the pin `8-11`,
from `/proc/stat` at the start of the energy phase to `/proc/stat` after the pair.
The same string is copied onto every meter row of that pair.
There is no 1 Hz series inside the trial window, and there is no per-trial maximum of such samples.
The `c82c1e96` mean criterion (mean of 1 Hz samples ≤ 0.05) cannot be evaluated.
The `c82c1e96` sample criterion (no 1 Hz sample above 0.15) cannot be evaluated.

The columns below compare those single cumulative fractions to 0.05 and 0.15 anyway.
A row that fails the comparison is not a trial the rule refused. Dropping it would be selection after the fact.
Later pairs include earlier pairs in the `/proc/stat` delta, so the fraction is not the trial window alone.

Of 636 attributable meter-trials, 204 have a cross-core mean of that one snapshot above 0.05,
and 138 have at least one outside core above 0.15.
Package, core, and NVML share the busy string. The counts are not three measurements.

| workload | side | scope | attributable | max core fraction | max cross-core mean | core > 0.15 | cross-core mean > 0.05 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| thrml_readme_ising_chain_n5 | ferrotherm | package | 9/10 | 0.0358 | 0.0029 | 0 | 0 |
| thrml_readme_ising_chain_n5 | ferrotherm | core | 10/10 | 0.0358 | 0.0029 | 0 | 0 |
| thrml_readme_ising_chain_n5 | ferrotherm | gpu | 9/10 | 0.0358 | 0.0029 | 0 | 0 |
| thrml_readme_ising_chain_n5 | thrml | package | 9/10 | 0.0358 | 0.0029 | 0 | 0 |
| thrml_readme_ising_chain_n5 | thrml | core | 10/10 | 0.0358 | 0.0029 | 0 | 0 |
| thrml_readme_ising_chain_n5 | thrml | gpu | 9/10 | 0.0358 | 0.0029 | 0 | 0 |
| ising_grid2d_open_3x3 | ferrotherm | package | 9/10 | 0.0317 | 0.0025 | 0 | 0 |
| ising_grid2d_open_3x3 | ferrotherm | core | 9/10 | 0.0317 | 0.0025 | 0 | 0 |
| ising_grid2d_open_3x3 | ferrotherm | gpu | 10/10 | 0.0365 | 0.0048 | 0 | 0 |
| ising_grid2d_open_3x3 | thrml | package | 10/10 | 0.0365 | 0.0048 | 0 | 0 |
| ising_grid2d_open_3x3 | thrml | core | 10/10 | 0.0365 | 0.0048 | 0 | 0 |
| ising_grid2d_open_3x3 | thrml | gpu | 10/10 | 0.0365 | 0.0048 | 0 | 0 |
| potts_path_q3_n6 | ferrotherm | package | 0/10 |  |  |  |  |
| potts_path_q3_n6 | ferrotherm | core | 0/10 |  |  |  |  |
| potts_path_q3_n6 | ferrotherm | gpu | 0/10 |  |  |  |  |
| potts_path_q3_n6 | thrml | package | 0/10 |  |  |  |  |
| potts_path_q3_n6 | thrml | core | 0/10 |  |  |  |  |
| potts_path_q3_n6 | thrml | gpu | 0/10 |  |  |  |  |
| tsp_iv4_three_body_d8 | ferrotherm | package | 0/10 |  |  |  |  |
| tsp_iv4_three_body_d8 | ferrotherm | core | 0/10 |  |  |  |  |
| tsp_iv4_three_body_d8 | ferrotherm | gpu | 0/10 |  |  |  |  |
| tsp_iv4_three_body_d8 | thrml | package | 0/10 |  |  |  |  |
| tsp_iv4_three_body_d8 | thrml | core | 0/10 |  |  |  |  |
| tsp_iv4_three_body_d8 | thrml | gpu | 0/10 |  |  |  |  |
| ising_open_4x4 | ferrotherm | package | 0/10 |  |  |  |  |
| ising_open_4x4 | ferrotherm | core | 0/10 |  |  |  |  |
| ising_open_4x4 | ferrotherm | gpu | 0/10 |  |  |  |  |
| ising_open_4x4 | thrml | package | 0/10 |  |  |  |  |
| ising_open_4x4 | thrml | core | 0/10 |  |  |  |  |
| ising_open_4x4 | thrml | gpu | 0/10 |  |  |  |  |
| ising_open_8x8 | ferrotherm | package | 6/10 | 0.2098 | 0.0795 | 6 | 6 |
| ising_open_8x8 | ferrotherm | core | 6/10 | 0.2098 | 0.0795 | 6 | 6 |
| ising_open_8x8 | ferrotherm | gpu | 6/10 | 0.2098 | 0.0795 | 6 | 6 |
| ising_open_8x8 | thrml | package | 6/10 | 0.2098 | 0.0795 | 6 | 6 |
| ising_open_8x8 | thrml | core | 6/10 | 0.2098 | 0.0795 | 6 | 6 |
| ising_open_8x8 | thrml | gpu | 6/10 | 0.2098 | 0.0795 | 6 | 6 |
| ising_open_16x16 | ferrotherm | package | 10/10 | 0.1901 | 0.0723 | 10 | 10 |
| ising_open_16x16 | ferrotherm | core | 10/10 | 0.1901 | 0.0723 | 10 | 10 |
| ising_open_16x16 | ferrotherm | gpu | 10/10 | 0.1901 | 0.0723 | 10 | 10 |
| ising_open_16x16 | thrml | package | 10/10 | 0.1901 | 0.0723 | 10 | 10 |
| ising_open_16x16 | thrml | core | 10/10 | 0.1901 | 0.0723 | 10 | 10 |
| ising_open_16x16 | thrml | gpu | 10/10 | 0.1901 | 0.0723 | 10 | 10 |
| ising_open_32x32 | ferrotherm | package | 10/10 | 0.1642 | 0.0627 | 7 | 10 |
| ising_open_32x32 | ferrotherm | core | 10/10 | 0.1642 | 0.0627 | 7 | 10 |
| ising_open_32x32 | ferrotherm | gpu | 10/10 | 0.1642 | 0.0627 | 7 | 10 |
| ising_open_32x32 | thrml | package | 10/10 | 0.1642 | 0.0627 | 7 | 10 |
| ising_open_32x32 | thrml | core | 10/10 | 0.1642 | 0.0627 | 7 | 10 |
| ising_open_32x32 | thrml | gpu | 10/10 | 0.1642 | 0.0627 | 7 | 10 |
| ising_open_64x64 | ferrotherm | package | 9/10 | 0.1452 | 0.0552 | 0 | 8 |
| ising_open_64x64 | ferrotherm | core | 9/10 | 0.1452 | 0.0552 | 0 | 8 |
| ising_open_64x64 | ferrotherm | gpu | 9/10 | 0.1452 | 0.0552 | 0 | 8 |
| ising_open_64x64 | thrml | package | 9/10 | 0.1452 | 0.0552 | 0 | 8 |
| ising_open_64x64 | thrml | core | 9/10 | 0.1452 | 0.0552 | 0 | 8 |
| ising_open_64x64 | thrml | gpu | 9/10 | 0.1452 | 0.0552 | 0 | 8 |
| potts_open_q3_4x4 | ferrotherm | package | 10/10 | 0.1334 | 0.0495 | 0 | 0 |
| potts_open_q3_4x4 | ferrotherm | core | 10/10 | 0.1334 | 0.0495 | 0 | 0 |
| potts_open_q3_4x4 | ferrotherm | gpu | 10/10 | 0.1334 | 0.0495 | 0 | 0 |
| potts_open_q3_4x4 | thrml | package | 10/10 | 0.1334 | 0.0495 | 0 | 0 |
| potts_open_q3_4x4 | thrml | core | 10/10 | 0.1334 | 0.0495 | 0 | 0 |
| potts_open_q3_4x4 | thrml | gpu | 10/10 | 0.1334 | 0.0495 | 0 | 0 |
| potts_open_q3_8x8 | ferrotherm | package | 10/10 | 0.1236 | 0.0449 | 0 | 0 |
| potts_open_q3_8x8 | ferrotherm | core | 10/10 | 0.1236 | 0.0449 | 0 | 0 |
| potts_open_q3_8x8 | ferrotherm | gpu | 10/10 | 0.1236 | 0.0449 | 0 | 0 |
| potts_open_q3_8x8 | thrml | package | 10/10 | 0.1236 | 0.0449 | 0 | 0 |
| potts_open_q3_8x8 | thrml | core | 10/10 | 0.1236 | 0.0449 | 0 | 0 |
| potts_open_q3_8x8 | thrml | gpu | 10/10 | 0.1236 | 0.0449 | 0 | 0 |
| potts_open_q4_4x4 | ferrotherm | package | 10/10 | 0.1151 | 0.0410 | 0 | 0 |
| potts_open_q4_4x4 | ferrotherm | core | 10/10 | 0.1151 | 0.0410 | 0 | 0 |
| potts_open_q4_4x4 | ferrotherm | gpu | 10/10 | 0.1151 | 0.0410 | 0 | 0 |
| potts_open_q4_4x4 | thrml | package | 10/10 | 0.1151 | 0.0410 | 0 | 0 |
| potts_open_q4_4x4 | thrml | core | 10/10 | 0.1151 | 0.0410 | 0 | 0 |
| potts_open_q4_4x4 | thrml | gpu | 10/10 | 0.1151 | 0.0410 | 0 | 0 |
| potts_open_q4_8x8 | ferrotherm | package | 10/10 | 0.1083 | 0.0379 | 0 | 0 |
| potts_open_q4_8x8 | ferrotherm | core | 10/10 | 0.1083 | 0.0379 | 0 | 0 |
| potts_open_q4_8x8 | ferrotherm | gpu | 10/10 | 0.1083 | 0.0379 | 0 | 0 |
| potts_open_q4_8x8 | thrml | package | 10/10 | 0.1083 | 0.0379 | 0 | 0 |
| potts_open_q4_8x8 | thrml | core | 10/10 | 0.1083 | 0.0379 | 0 | 0 |
| potts_open_q4_8x8 | thrml | gpu | 10/10 | 0.1083 | 0.0379 | 0 | 0 |
| tsp_iv4_three_body_d12 | ferrotherm | package | 7/10 | 0.1027 | 0.0352 | 0 | 0 |
| tsp_iv4_three_body_d12 | ferrotherm | core | 7/10 | 0.1027 | 0.0352 | 0 | 0 |
| tsp_iv4_three_body_d12 | ferrotherm | gpu | 7/10 | 0.1027 | 0.0352 | 0 | 0 |
| tsp_iv4_three_body_d12 | thrml | package | 7/10 | 0.1027 | 0.0352 | 0 | 0 |
| tsp_iv4_three_body_d12 | thrml | core | 7/10 | 0.1027 | 0.0352 | 0 | 0 |
| tsp_iv4_three_body_d12 | thrml | gpu | 7/10 | 0.1027 | 0.0352 | 0 | 0 |
| tsp_iv4_three_body_d16 | ferrotherm | package | 5/10 | 0.0967 | 0.0324 | 0 | 0 |
| tsp_iv4_three_body_d16 | ferrotherm | core | 5/10 | 0.0967 | 0.0324 | 0 | 0 |
| tsp_iv4_three_body_d16 | ferrotherm | gpu | 5/10 | 0.0967 | 0.0324 | 0 | 0 |
| tsp_iv4_three_body_d16 | thrml | package | 5/10 | 0.0967 | 0.0324 | 0 | 0 |
| tsp_iv4_three_body_d16 | thrml | core | 5/10 | 0.0967 | 0.0324 | 0 | 0 |
| tsp_iv4_three_body_d16 | thrml | gpu | 5/10 | 0.0967 | 0.0324 | 0 | 0 |
