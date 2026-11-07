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

Joules come from a hardware counter or they are null. Intel RAPL (via
ferrotherm-meter), the NVIDIA Management Library energy counter, or on macOS
the rootless `macmon` reader. No projected device price is consulted.
`results/report.json` in this tree is one run of that binary. Its `provenance`
object says which machine, which counters were absent, and that the tree was
or was not dirty. Read that object before quoting a number from the file.
