# Framework-only --scale CPU run

See TABLES.md first: this run's high contended baseline produced negative incremental-energy prices. Preserve them but do not claim energy superiority.

Framework machineId 94655143-44cb-4c99-a72a-a988f4af51e9 only. No calls targeted the Mac or thor.local in this task. Exact revision and pinned runtime are in provenance and audit.json. The package and core counters are AMD RAPL on this Ryzen AI Max+ 395: model- or estimate-based SoC energy, not wall power, and no DRAM domain (Schöne et al., IEEE CLUSTER 2021, arXiv:2108.00808; Zen 5 / Strix Halo not validated by a primary measurement study; `amd_energy` says Measured and does not say how). That label does not make the negative increments below a wall-checked price, and it is not Intel RAPL. Core is a subdomain and is not added to package energy. Receipt numbers are unchanged. Both recorded maximum ranges are 65532610987 microjoules. Wrap-safe raw readings saved at 10 Hz.

Commands: existing user Rust build; JAX_PLATFORMS=cpu ferrotherm-thrml-bench --scale --device cpu --trials 5 --python ~/bench-scratch/thrml-venv/bin/python, first quiet, then --contended because quiet returned load_gated. All 15 workloads and 150 measured trials finished exit 0; no timeout. All long work used nohup and scratch logs. No sudo, configuration changes, code modifications, pushes, PR, merge, deployment or other-user process termination. No benchmark processes remain.

The external driver bounds no-progress intervals at 180 s and total invocation at 1800 s. The pinned harness has no single-workload selector or per-cell timeout; if it had stalled, only its own process group would have been stopped and downstream work marked not-run. No watchdog fired.

Energy summaries, J/ESS CIs, attribution flags and accuracy error CIs are copied unchanged. Raw arithmetic quotients and mean-energy reference coverage are derived from the serialized records with formulas specified in TABLES.md, not newly measured quantities. Missing/negative measurements were not replaced or clamped. Historical evidence remains separate.


Provenance dirty flag: true. Final git status identifies only generated untracked bench/thrml-tsp/python/__pycache__/platform_choice.cpython-313.pyc. git diff --quiet HEAD returned 0; tracked source files match the pinned revision. Bytecode was left intact, not concealed. See final-tree-status.txt.
quiet start: 2026-11-07 07:48:29 EST
quiet end: 2026-11-07 07:50:00 EST
contended start: 2026-11-07 07:50:00 EST
contended end: 2026-11-07 07:57:15 EST
