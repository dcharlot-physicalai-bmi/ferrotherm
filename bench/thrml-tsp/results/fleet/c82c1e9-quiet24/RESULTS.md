# jetson-hub attempt 037

Intel RAPL core-domain J/ESS on jetson-hub vs THRML 0.1.4 on JAX CPU, per workload. Tarball sha256 1df27d9f6716ca6ad0da9a0fe4c773dfc8f5a46609e5b938cc086a00d7f1b613. Publish commit 22d3e720714c76e73389998d36e5b8159e89fe96.

These are the small accuracy-suite workloads, from the 5-spin chain through the open 64×64 (n=4096). This is not a large-grid energy claim. The open 64×64 core-domain quotient is 8.952 ± 0.762, about 9×, on that lattice only. `ising_open_4x4` is under investigation and is not a cited quotient.

## Receipt

Harness `c82c1e96c24d6f6c292af1196cd72199b0838257` on jetson-hub (13th Gen Intel Core i9-13900H), pinned to cpus 8–11. Attempt 037. Top grade, preflight, and energy are `quiet`. Accuracy is `measured`. The bundle grade is the energy grade.

The preflight held for 60.013 s. Maximum load1 was 1.47, under the unchanged threshold of 2. The host lock `/tmp/bmi-energy.lock` was held (`wait_s` 0, holder pid 3679283) from 2026-11-10T08:37:14.871Z to 2026-11-10T11:36:35.973Z.

900 trials, 15 workloads, 10 seeds each side. Intel RAPL core: 300 of 300 trials attributable, all paired. Intel RAPL package: 300 of 300 attribution refused. NVML (RTX 4050 Laptop GPU): 300 of 300 attribution refused. Package and NVML have gross and incremental joules and no J/ESS.

THRML 0.1.4 ran on CPU (`thrml=0.1.4`, `jax=0.12.0.1`, `jaxlib=0.12.0.1`, `equinox=0.13.8`, `device=cpu:0`, `JAX_PLATFORMS=cpu`).

These counters are Intel RAPL package and core domains. They are not wall power. Core J/ESS is not package energy and not whole-system energy. This is not the AMD model basis.

Attempts 001–036 are the refusal receipts in `jetson-hub/attempt-037/`, with `attempt-table.csv`. They are not quiet-host joules. `MANIFEST.txt` is the tarball manifest. Paths it marks `LEFT_OUT` are not reconstructed.

## Tables

Per-workload core J/ESS, the propagated quotient, and the refused package and NVML joules are in `TABLES.md`. The uncertainty terms are in `jetson-hub/attempt-037/UNCERTAINTY.md`.

Combined uncertainty is the root-sum-square of the receipt's Student-t half-width, quantization, cadence/edge, and bracket terms. The quotient is THRML core J/ESS divided by ferrotherm core J/ESS. Its uncertainty is `R * sqrt((u_t/J_t)^2 + (u_f/J_f)^2)` with those combined terms. Meters are not pooled.

## `ising_open_4x4`

The ferrotherm core mean is `1.22954420e-4`, above the open 8×8 mean `5.19702797e-5`. The row stays marked under investigation. The cause is in `UNCERTAINTY.md`: each metered repetition enumerates the 2^16 law because `exact_energy` is unset for side ≤ 4, while the 8×8 scores the transfer-matrix mean. Energy-window ESS is higher on the 4×4 (1273 vs 1045) and `tau_int` is lower (0.780 vs 1.087), so the integrated autocorrelation is not the cause. The 4×4 repetition takes 7.92 ms and 0.157 core-joules; the sampler timer alone, which stops before the score, records 0.88 ms. The 8×8 sampler timer (2.59 ms) and the metered repetition (2.67 ms) agree. The same window on all 15 workloads is audited in `jetson-hub/attempt-037/SCORING-AUDIT.md`. That audit does not change these grades.

## Package and NVML

Every package trial drifted by 1.438–2.592 W (mean 1.967 W), and every one exceeded 3 times its own idle-bracket sigma. The NVML drift series is the same 300 numbers: those rows store the package idle bracket. Core drift was 0.0003–0.249 W, and 0 of 300 core trials exceeded 3σ. The published grades are unchanged. A later rule is proposed in `docs/proposals/bracket-drift-v2.md` and is not applied here.
