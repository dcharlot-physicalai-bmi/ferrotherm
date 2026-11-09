# Fixed-window fleet rerun

Both hosts used exact 0029c270ea401109254f691ad78bc1666c9540df, clean checkout, user Rust, user venv THRML 0.1.4/JAX 0.12.0.1 CPU. All long operations used nohup under ~/bench-scratch. No sudo, configuration changes, repository-content modifications, merge or deployment. All benchmark and recorder processes finished.

Framework registered ID 94655143-44cb-4c99-a72a-a988f4af51e9, hostname framework-128gb-0000. The package and core counters are AMD RAPL: model- or estimate-based SoC energy, not wall power, and this interface has no DRAM domain. Schöne et al., IEEE CLUSTER 2021 (arXiv:2108.00808), on a Zen 2 EPYC 7502, found that the energy data is modeled, not measured. Zen 5 / Strix Halo accuracy on this Ryzen AI Max+ 395 has not been validated by any primary measurement study. The kernel `amd_energy` hwmon text calls the inputs Measured and does not say how; that is not a claim that AMD documentation says the counters are modeled. Absolute J/ESS here needs a wall-meter cross-check before it can be claimed. Same-host ratios in `RESULTS.md` may stand with that basis stated. They are not Intel RAPL. Rootless package/core counters readable; max_energy_range_uj=65532610987 for each; wrap-safe external traces preserved. Core is a package subdomain: not added to package energy. Quiet mode tried first; load_gated, then authorized --contended. The receipt bytes were not rewritten.

M5 Max 48GB registered ID 25022ed7-76ae-44a9-8b17-201eeee0fcdc, rootless macmon 0.10.0, SoC telemetry (`sys_power`), not wall power. The receipt scope token is `whole-system` and names that field. Secondary evidence, explicit --contended. Initial load1 was below 2 but the explicit mode remains contended host; it was not promoted to quiet evidence.

Each measured run: --trials 5, all four canonical workloads, alternating ferrotherm/THRML. Every trial uses >=2-second actual windows with repetitions. Reported gross and incremental J, seconds and ESS are normalized per sampler repetition. window_seconds and repetitions preserve actual measurement-window context. Do not compare the new normalized joules directly with the prior revision's unnormalized repeated-window totals.

TABLES.md copies the harness per-workload summaries exactly, including 95% CIs; no confidence intervals were recomputed, no workloads reconstructed, and no trials excluded. audit.json records window/update checks, attribution and load ranges. All 80 measured trials were marked attributable, each had >=8 counter updates, no gross<=0 or incremental<0 values, and no short windows. This validates those harness checks, not a quiet-host or hardware-silicon energy-leadership claim. The four measured models are a 5-spin chain, an open 3×3 (9 spins), a 6-site Potts path, and an 8-spin three-body system. `RESULTS.md` quotes the package span 529×–1703× and the SoC-telemetry span 347×–1218× only as the endpoints of those four rows. They are not a result on a larger grid.

Caveats: host contention is present and grades are not pooled with quiet mode. The above-baseline reference in a contended run is not a certified quiet idle measurement. The Student-t intervals in the harness cells are trial scatter. `RESULTS.md` adds the quantization, cadence-edge, and baseline-bracket terms beside them and states what the raw traces could and could not recompute. RAPL package and macmon SoC telemetry stay separate, and the core counter in the raw trace is not added to package. Report provenance contains legacy generic fleet-hardware paragraphs; current energy fields and trial backend/scope are authoritative for this run. Compilation emitted the pre-existing suspicious_runtime_symbol_definitions warning for the extern read signature; build succeeded without suppression or code changes. No additional measured-trial anomalies were detected.

Raw counter traces, all reports, accuracy traces, logs, exact driver and hardware/software provenance are retained under host directories. Historical prior-revision archive is included separately and must not be pooled with this revision.

framework quiet start: 2026-11-07 07:27:43 EST
framework quiet end: 2026-11-07 07:28:27 EST
framework contended start: 2026-11-07 07:28:27 EST
framework contended end: 2026-11-07 07:30:13 EST
m5 start: 2026-11-07 07:27:42 EST
m5 end: 2026-11-07 07:29:35 EST
