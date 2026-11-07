# Fleet benchmark attempt — 7 November 2026

No attributable benchmark joules were obtained. No energy superiority claim is supported.

| Target | Gross J | Incremental J | J/effective sample | 95% energy CI | Outcome |
|---|---|---|---|---|---|
| Framework registered primary ID | unavailable | unavailable | unavailable | unavailable | ID resolves to framework-32gb-0001, not requested framework-128gb-0000; all RAPL energy_uj reads permission-denied; unprivileged perf refused, paranoid=4 |
| M5 Max 48GB | unavailable | unavailable | unavailable | unavailable | Rootless macmon works; pinned harness refused idle baseline: load1=6.45 > 2 |

## M5 Max
- Original existing checkout untouched. Isolated clone under ~/bench-scratch/ferrotherm-thrml, detached at e92c9fc1a263ad52862105e05175ef83e2ba7796. The remote branch had advanced to 263e62b7028af8021215ebf546ffc6f7bf1a1ad0; the requested exact commit was used.
- User venv ~/bench-scratch/thrml-venv, Python 3.13; pinned THRML 0.1.4 / JAX 0.12.0.1 CPU. See mac-provenance.txt for complete package versions.
- User Rust toolchain; no sudo or system config changes.
- Build and run used nohup background commands with logs under ~/bench-scratch.
- Harness command: JAX_PLATFORMS=cpu target/release/ferrotherm-thrml-bench --out ~/bench-scratch/m5-max-results --trials 5 --python ~/bench-scratch/thrml-venv/bin/python
- Run completed exit 0 at 06:52:25–06:52:41 EST. Four workloads, five accuracy trials each for both samplers.
- macmon 0.10.0 rootless raw traces: 30 samples before, 146 samples during, 30 samples after. These external traces include mixed execution/JIT and are not per-sampler energy estimates. No integration into benchmark joules was claimed.
- The pinned harness never meters the scored chains and leaves their joules null. Its separate >=2-second energy repeat covers ferrotherm only, not THRML, and refuses to divide by the accuracy-chain effective sample size. Even on a quiet host this harness does not implement the requested five-trial head-to-head energy comparison.
- No stray benchmark processes found before run or after completion. No processes belonging to other work were stopped.

## Publication status
No commit or push performed. Repository changes must be handed to a CloudAgent; that tool is absent from this executor's available tool catalog. The parent can attach this evidence bundle to an authorized cloud agent, ask it to preserve these limitations, and commit/push results to bench/thrml-tsp-fleet without opening a PR.
Compare target (not evidence of a new push): https://github.com/dcharlot-physicalai-bmi/ferrotherm/compare/main...bench/thrml-tsp-fleet
