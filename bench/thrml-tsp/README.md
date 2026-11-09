# THRML head-to-head

Accuracy of ferrotherm against Extropic's THRML™ 0.1.4 on four pinned workloads,
scored with one estimator. The comparison write-up is
`docs/prior-art/thrml-and-tsp.md`.

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -r bench/thrml-tsp/python/requirements.txt
cargo run --release -p ferrotherm-thrml-bench -- \
  --out bench/thrml-tsp/results \
  --python "$(which python)"
```

`requirements.txt` pins the resolution that was smoke-tested on 7 November 2026:
THRML™ 0.1.4, JAX 0.12.0.1 with the CPU extra, equinox 0.13.8. JAX is not a
dependency THRML™ declares. The pin is this harness's pin.

The binary refuses fewer than two trials. Default schedule: 5 trials, 400
warmup sweeps, 2000 draws, thin 1. THRML™'s warmup is set to `warmup + thin`
so the first kept state matches `gibbs::Sampler::collect`.

Joules come from a hardware counter or they are null. The quiet-host gate is
unchanged: a 1-minute load average above 2 is not quiet. `--contended` is a
different mode. It still subtracts idle and reports 95% intervals, and every
figure from that run is evidence grade `contended host`. Those figures are not
quiet-host evidence and are not pooled with a quiet run. The schedule is
A/B/A: ferrotherm, THRML, ferrotherm, and so on, five or more of each. Each
trial repeats the sampler until the window is at least 2 s, divides joules and
effective-sample size by the repetition count, and records `load1`,
other-process CPU, the repetition count, and how many times the counter
advanced. Fewer than 8 counter updates is not attributable. A zero gross with a
negative incremental is that case.

Instruments, one per trial: rootless macmon (`sys_power`, SoC telemetry, not
wall power) or an INA3221 rail (board) when that backend is open; otherwise
RAPL `energy_uj` (package, unless the honest domain is `psys`) with the core
domain reported separately and not added; otherwise `perf` event
`power/energy-pkg` (package); otherwise NVML (gpu). Summaries are per workload,
sampler, instrument, and scope. Package joules, core joules, and SoC-telemetry
joules are not added together. RAPL on hostname `framework-128gb-0000` is the
primary energy evidence. That host reads package `energy_uj` without root.
macmon is secondary.

`--quiet-gate` samples load1 once a second for 60 seconds, and the busy
fraction of cores outside `--cpus` over that same window, before any workload
runs. The run starts only when every load sample is at or under 2. The busy
fraction is stamped on the receipt and is not a second threshold. The evidence
grade is written on the report, on each workload, and on `quiet-gate.json`.
`--quiet-gate` and `--contended` are different grades and are refused together.
When that preflight holds, the energy phase runs next and the accuracy phase
follows it. load1 is a one-minute average, so accuracy and THRML before the
energy check raise the average the check then refuses. The threshold stays 2.
No process is taken out of the load. The energy section still requires load1
at or under 2. `settle_load1` on the receipt is null: energy used the passing
preflight and did not wait for a second 60 s decay.

`--scale` adds the larger workloads: open Ising at 4×4, 8×8, 16×16 (transfer
matrix; 4×4 is also enumerated), 32×32 and 64×64 (Onsager's infinite lattice,
not the finite mean), q=3 and q=4 Potts grids with a transfer-matrix mean, and
three-body models at d=12 and d=16. `--device cuda` asks the THRML runner for a
JAX GPU and exits if JAX has none. It does not run the CPU device under that
name. `--device auto` uses a GPU when JAX has one.

## Fleet evidence, 7 November 2026

`results/fleet/` keeps two records.

The first record, the files directly under `results/fleet/` other than
`189502d/`, obtained no joules.

- M5 Max 48 GB, rootless macmon 0.10.0. Accuracy trials finished. The harness
  refused the idle baseline because load1 was 6.45, above 2. The macmon traces
  are secondary and contended. They were not integrated into joules, and they
  are not quiet-host evidence.
- The host that answered that Framework request was `framework-32gb-0001`.
  Every RAPL `energy_uj` read there was permission denied, and
  `perf stat -e power/energy-pkg/` failed with `perf_event_paranoid` 4.
  That denial is `framework-32gb-0001`. It is not a property of
  `framework-128gb-0000`.

`results/fleet/189502d/` is the first measured-joules bundle, revision
`189502d`, both hosts evidence grade `contended host`.

- `framework-128gb-0000` (AMD RYZEN AI MAX+ 395) reads RAPL package
  `energy_uj` without root. Quiet mode stayed above load 2 and recorded
  `load_gated` with no energy trials. `--contended` then measured 40 trials.
  Scope is CPU package. Core is a subdomain and is not added to package energy.
- M5 Max 48 GB, rootless macmon 0.10.0, `--contended`, 40 trials. The scope is
  SoC telemetry (`sys_power`), not wall power. Those joules are not combined
  with the package joules.
- The ferrotherm RAPL windows in that bundle are about 1 ms. One trial records
  gross 0 J, incremental about −0.031 J, and `attributable: true`. Those values
  stay as measured. They are below the counter cadence. The harness after that
  revision does not attribute a window with fewer than 8 counter updates.

`results/fleet/0029c27/` is the fixed-window rerun of that harness. 40 of 40
trials on each host were attributable, evidence grade `contended host`. RAPL
package and macmon SoC telemetry stay in separate tables. The per-workload
J/ESS intervals and the limits are in `results/fleet/0029c27/RESULTS.md`.
There is no quiet-host leadership claim in that file.

`results/fleet/8e9ad0a-scale/` is a later `--scale` run on `framework-128gb-0000`.
It is **invalid for energy**: one 144.8 W baseline was subtracted while load1 moved
from about 11 to about 24, and negative increments were marked attributable. The raw
traces stayed on that machine. See `INVALID.md` there.

`results/report.json` is an earlier counter-less accuracy run on a Xeon guest.
Its provenance says which counters were absent. Do not read a joule out of it.
