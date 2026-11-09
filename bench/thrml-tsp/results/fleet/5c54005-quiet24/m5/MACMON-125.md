# macmon pull 125

macmon 0.11.0 (2026-11-06, pull request 125, commit `8c732646`) drops `ram_power` values above system power. The change does not alter `sys_power`, which is `max(smc_sys, cpu+gpu+ane)`.

Recorded macmon version: **0.11.0**, from `provenance.txt`. The host line in that file is macOS 27.2. This binary is the one that contains the fix. The M5 is on macmon 0.11.0 now.

Trace: `attempt-002-macmon-raw.jsonl`, 2015 samples. The macmon object is nested in the `macmon` string. Every sample has `ram_power` and `sys_power`. Maximum `ram_power` is 0. Maximum `sys_power` is 82.22364807128906 W. Samples with `ram_power` greater than `sys_power`: 0. Samples with `ram_power` above 100 W: 0.

Attempt 1 has no macmon raw trace. Attempt 2's energy outcome is unavailable: the idle baseline was refused, `idle_watts` is null, and no trial joules were stored. The harness reads `sys_power` only. `ram_power` is not an input. These joules, of which there are none, are unaffected. Receipt numbers were not rewritten.
