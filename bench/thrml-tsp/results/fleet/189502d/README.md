# Run notes and limits

Revision 189502d509080634a1fbe3253876c1b75e5bbd56; both reports record a clean checkout. THRML 0.1.4 and JAX 0.12.0.1 CPU, user-level venv/toolchain. All long work ran with nohup under ~/bench-scratch. No sudo, system changes, merge, deployment, or repository-content changes.

Framework: registered machine 94655143-44cb-4c99-a72a-a988f4af51e9, hostname framework-128gb-0000, AMD RYZEN AI MAX+ 395. RAPL is the primary instrument, CPU package scope. Rootless readable package/core counters; both max_energy_range_uj=65532610987. Core is a package subdomain and is NOT added to package energy. Quiet-mode load samples stayed above 2: [2.68, 2.68, 2.62, 2.62, 2.57, 2.57, 2.57, 2.53, 2.53, 2.48, 2.48, 2.48, 2.44, 2.44, 2.41]. Quiet returned load_gated with zero energy trials; then --contended returned measured/contended host, 40 trials (five of each sampler per workload).

M5 Max 48GB: registered machine 25022ed7-76ae-44a9-8b17-201eeee0fcdc. Rootless macmon 0.10.0, secondary instrument, whole-system scope. --contended returned measured/contended host, 40 trials.

Numbers are the unmodified harness output; per-workload means/CIs derive from serialized trial values and may differ at the last rounding digit from a computation on the harness's unrounded internal floats. Each workload has a fixed ten-row alternating ferrotherm/THRML schedule. The harness omits workload IDs on energy rows and pools all workloads in its top-level energy summaries. The supplied derived table instead reconstructs the known canonical order and asserts every side/trial index. Reproduction: python3 derive-summary.py. All CIs are two-sided 95% Student-t for five observations, t4=2.7764451052. They describe trial scatter, not a full uncertainty budget for baseline drift or host contention.

Critical caveats:
- Both measured runs have evidence grade contended host. Neither is quiet-host evidence.
- A busy-host no-load reference is subtracted; this is not a certified quiet idle baseline. No contaminated result is silently excluded.
- Framework ferrotherm windows span only 0.000783–0.001082 seconds; THRML windows span 0.270585–1.209886 seconds. One ferrotherm chain trial records gross_j=0, incremental_j=-0.0310099424, joules_per_ess=-1.79477212e-5 and attributable=true. These exact values remain in the bundle. Counter update cadence/resolution needs validating before a leadership claim based on these tiny windows.
- M5 ferrotherm windows are approximately 1.5 seconds; THRML windows approximately 1.55–1.99 seconds. Cross-host gross joules must not be compared as equal-scope instruments.
- J/ESS is the harness's incremental joules divided by the effective sample size of the measured execution, not gross J/ESS. Negative values and confidence limits are retained, not clamped.
- The legacy provenance paragraphs contain stale generic statements about missing fleet hardware; use the current energy backend fields, raw traces and this run's actual trial records instead.
- Build succeeded with a suspicious_runtime_symbol_definitions warning about the extern read signature. Build logs are preserved; no warning suppression or code changes were made.

Raw traces: Framework raw RAPL values include per-reading wrap-corrected deltas and ranges. M5 raw macmon JSONL records are retained separately. Neither external whole-run trace is substituted for the harness's per-trial measurements. Reports include trial load1, other-process CPU when available, ESS, duration, gross/incremental J and grade/rank. No benchmark processes remained after completion.

framework quiet start: 2026-11-07 07:13:46 EST
framework quiet end: 2026-11-07 07:14:31 EST
framework contended start: 2026-11-07 07:14:31 EST
framework contended end: 2026-11-07 07:15:04 EST
m5 start: 2026-11-07 07:13:49 EST
m5 end: 2026-11-07 07:15:20 EST
