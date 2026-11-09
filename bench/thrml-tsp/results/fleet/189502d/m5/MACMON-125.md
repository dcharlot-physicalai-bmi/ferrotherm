# macmon pull 125

macmon 0.11.0 (2026-11-06, pull request 125, commit `8c732646`) drops `ram_power` values above system power. The change does not alter `sys_power`, which is `max(smc_sys, cpu+gpu+ane)`.

Recorded macmon version: **0.10.0**, from `evidence-189502d/provenance.txt`. That build is older than the fix.

Trace: `evidence-189502d/macmon-raw.jsonl`, 802 samples. Every sample has `ram_power` and `sys_power`. Maximum `ram_power` is 0. Maximum `sys_power` is 46.7645263671875 W. Samples with `ram_power` greater than `sys_power`: 0. Samples with `ram_power` above 100 W: 0.

The harness joules for this run are mean `sys_power` times duration. `ram_power` is not an input. This directory's joules are unaffected. Receipt numbers were not rewritten.
