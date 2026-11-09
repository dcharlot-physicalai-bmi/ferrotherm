# macmon pull 125

macmon 0.11.0 (2026-11-06, pull request 125, commit `8c732646`) drops `ram_power` values above system power. The change does not alter `sys_power`, which is `max(smc_sys, cpu+gpu+ane)`.

These four files sit in this directory, not under an `m5/` receipt. `SUMMARY.md` records them as macmon **0.10.0** rootless traces and says they were not integrated into benchmark joules. The jsonl files themselves do not name a version.

| File | Samples | `ram_power` > `sys_power` | `ram_power` > 100 W | max `ram_power` | max `sys_power` |
| --- | --- | --- | --- | --- | --- |
| `macmon-probe.jsonl` | 3 | 0 | 0 | 0 | 16.56009292602539 |
| `macmon-idle-before.jsonl` | 30 | 0 | 0 | 0 | 15.498096466064453 |
| `macmon-run.jsonl` | 146 | 0 | 0 | 0 | 41.49776840209961 |
| `macmon-idle-after.jsonl` | 30 | 0 | 0 | 0 | 36.831417083740234 |

Every sample has both fields. No sample is affected. No joule was computed from `ram_power`. Unaffected. Receipt numbers were not rewritten.

Per-receipt scans for the later M5 bundles are `0029c27/m5/MACMON-125.md`, `189502d/m5/MACMON-125.md`, `5c54005-quiet24/m5/MACMON-125.md`, and `m5-max-results/MACMON-125.md`.
