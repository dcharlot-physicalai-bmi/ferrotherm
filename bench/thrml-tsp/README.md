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
trial records `load1` and other-process CPU time.

Instruments, one per trial: rootless macmon or an INA3221 rail when that
backend is open; otherwise RAPL `energy_uj`; otherwise `perf` event
`power/energy-pkg`; otherwise NVML. NVML is not added to a CPU counter.
RAPL on hostname `framework-128gb-0000` is the primary energy evidence.
macmon is secondary.

## Fleet evidence, 7 November 2026

No joules were obtained. No energy comparison is supported by that bundle.

`results/fleet/` is the measurement as it came off the machines.

- M5 Max 48 GB, rootless macmon 0.10.0. Accuracy trials finished. The harness
  refused the idle baseline because load1 was 6.45, above 2. The macmon traces
  are secondary and contended. They were not integrated into joules, and they
  are not quiet-host evidence.
- The Framework host that answered was `framework-32gb-0001`, not the requested
  `framework-128gb-0000`. Every RAPL `energy_uj` read was permission denied.
  `perf stat -e power/energy-pkg/` failed with `perf_event_paranoid` 4. No RAPL
  joules exist. Primary energy evidence was not collected.

`results/report.json` is an earlier counter-less accuracy run on a Xeon guest.
Its provenance says which counters were absent. Do not read a joule out of it.
