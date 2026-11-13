# c82c1e96 quiet-host joules, jetson-hub attempt 037

Intel RAPL core-domain J/ESS on jetson-hub vs THRML 0.1.4 on JAX CPU, per workload. Tarball sha256 1df27d9f6716ca6ad0da9a0fe4c773dfc8f5a46609e5b938cc086a00d7f1b613. Publish commit 22d3e720714c76e73389998d36e5b8159e89fe96.

`RESULTS.md` and `TABLES.md` are this publication. The receipt, the refusal receipts for attempts 001–036, the attempt table, and the uncertainty budget are in `jetson-hub/attempt-037/`.

Core is the only scope with J/ESS. Package and NVML joules are published without J/ESS. `ising_open_4x4` is under investigation. The open 64×64 core-domain quotient is about 9× on that lattice only. This is not a large-grid energy claim.

Attempt 037 stays as measured. Scoring ran inside the metered energy window on both sides. That defect is in ferrotherm's disfavour. `jetson-hub/attempt-037/SCORING-AUDIT.md` lists the affected workloads. The published means are not recomputed.

NVML for attempt 037 is ungraded. The traces are chain energies and states, not raw NVML idle samples, so there is no `regrade-v3 (per-meter bracket)` beside the as-measured rows. Those rows stay the package-bracket refusal: gross and incremental joules, no J/ESS.
