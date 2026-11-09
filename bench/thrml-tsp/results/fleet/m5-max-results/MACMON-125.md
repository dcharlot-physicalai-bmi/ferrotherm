# macmon pull 125

macmon 0.11.0 (2026-11-06, pull request 125, commit `8c732646`) drops `ram_power` values above system power. The change does not alter `sys_power`, which is `max(smc_sys, cpu+gpu+ane)`.

Recorded macmon version: none. `report.json` does not name a macmon version.

This directory has no macmon sample file, so there is no sample with `ram_power` greater than `sys_power` and none with `ram_power` above 100 W.

`energy.long_run` is `evidence: unavailable` with null joules (`load1` 6.45, above 2). `joules_per_effective_sample` is null. No joule in this receipt was taken from `ram_power`. Unaffected. Receipt numbers were not rewritten.
