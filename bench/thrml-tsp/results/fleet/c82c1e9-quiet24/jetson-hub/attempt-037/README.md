# Attempt 037

Intel RAPL core-domain J/ESS on jetson-hub vs THRML 0.1.4 on JAX CPU, per workload. Tarball sha256 1df27d9f6716ca6ad0da9a0fe4c773dfc8f5a46609e5b938cc086a00d7f1b613. Publish commit 22d3e720714c76e73389998d36e5b8159e89fe96.

`attempt-037/` is the quiet-host receipt (report, quiet gate, spec, and the traces that were in the tarball). `attempt-001` through `attempt-036` are refusal receipts. `attempt-table.csv` and `jetsonhub-c82c1e9-attempt-table.csv` list attempts 001–037. `attempt-037-j-per-ess.csv` is the per-workload summary extract. `jetsonhub-c82c1e9-quiet-attempt-037-publish.tar.gz.sha256` records the tarball hash. `MANIFEST.txt` is verbatim. Paths it marks `LEFT_OUT` are not in this tree.

`UNCERTAINTY.md` is the core J/ESS table, the refused package and NVML joules, and the 4×4 finding. `SCORING-AUDIT.md` lists every workload whose score ran inside the metered window. That defect is in ferrotherm's disfavour. The published means stay as measured.

NVML for attempt 037 is ungraded. Raw NVML idle samples are not in the traces, so there is no `regrade-v3 (per-meter bracket)` next to the as-measured grade. The as-measured NVML rows remain the package-bracket refusal.

The campaign narrative is `../../RESULTS.md`. The scan tables are `../../TABLES.md`.
