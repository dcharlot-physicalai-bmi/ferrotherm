# v3.1 launch amendment: JAX_PLATFORMS shim

bmi-0 approved this launch shim as a launch amendment to v3.1, on condition
this note is committed before any grade is read. The record is
`ferrotherm-v3.1-shim` `SUMMARY.md` and `SHA256SUMS`, collected 2026-11-20
at about 9:05 AM ET. `SUMMARY.md` sha256 is
`778f08ebd869db20b55540206a9b9e7ee4190da2aeb76355cf4992c8ec3c7ad8`.
The harness is `harness/v3.1` at
`4b9aa4da02142efb58605b4c67cfa4a2191c1b16`, unmodified. The shim is passed
as `--python` and `--python-b`. It installs nothing and changes no package
or version.

The copies live in `docs/launch-amendments/v3.1-shim/`. Their bytes are the
recorded files. `SHA256SUMS` in that directory is the recorded sums file.

## What the shim does

It reads the token after `--device` in its arguments.

- `--device cpu` sets `JAX_PLATFORMS` to `cpu`.
- `--device cuda` sets `JAX_PLATFORMS` to `cuda`.
- Any other value, or no `--device`, sets nothing.

It sets nothing else. It then runs `exec <venv>/bin/python "$@"`. It does
not change `runner.py`, THRML, or JAX. It does not move the 0.05 parity
line or any other v3.1 rule.

## Why

`bench/thrml-tsp/python/runner.py` `bind_device()`, on `777f0417` and on
v3.1, calls `jax.devices()`, which lists only the default backend. Inside a
CUDA venv that list is `[CudaDevice(id=0)]`, so the CPU suite (`--device
cpu`, the `--campaign` CPU half) fails with `cpu was requested and JAX has
no CPU device`. v3 jetson-hub attempt-015 logged that line 24,006 times.
`jax.devices("cpu")` in those venvs is `[CpuDevice(id=0)]`.

## The four files

| Host | Path | sha256 | Wraps |
| --- | --- | --- | --- |
| jetson-hub | `docs/launch-amendments/v3.1-shim/jetson-hub/python-A` | `a9d87a4a2c9c93545ddd606b64fc2cba15114a96fb15ece58349260d2b4aedc0` | `/home/dcharlot/bench-scratch/thrml-0.1.4-cuda/bin/python` (THRML A 0.1.4, JAX 0.12.0.1 CUDA13 plugin). mtime 2026-11-20 08:14:31 ET |
| jetson-hub | `docs/launch-amendments/v3.1-shim/jetson-hub/python-B` | `ae99480202869ffd60d98e755d42b363d2b9a81df6b41f85cfc4fbadeacbe4c9` | `/home/dcharlot/bench-scratch/thrml-main-2d96f4b4-cuda/bin/python` (THRML B git `2d96f4b4`, JAX 0.12.0.1). mtime 2026-11-20 08:14:31 ET |
| thor.local | `docs/launch-amendments/v3.1-shim/thor.local/python-A` | `26a4df233ffc1a9977c0740aa30bc4693c81a3e216eeb6c96fc07852c1244b0d` | `/home/cbio/bench-scratch/thrml-venv-cuda/bin/python` (THRML A 0.1.4, JAX 0.12.0.1). mtime 2026-11-20 08:19:17 ET |
| thor.local | `docs/launch-amendments/v3.1-shim/thor.local/python-B` | `ec547d0bba4f78e779d528d6b2b809cbe5e33ef0837ffdb4ca7f3f186f3f0388` | `/home/cbio/bench-scratch/thrml-main-2d96f4b4-cuda/bin/python` (THRML B git `2d96f4b4`, JAX 0.12.0.1). mtime 2026-11-20 08:19:17 ET |

## Interpreters

On jetson-hub, `bin/python3` in both of those venvs was repointed at
2026-11-20 08:13 ET, before any v3.1 attempt, to uv CPython 3.12.3 at
`/home/dcharlot/.local/share/uv/python/cpython-3.12.3-linux-x86_64-gnu/bin/python3.12`,
sha256 `4933b6c4a8521fb3aa93856701e30b3ac2626d3c47ceb22965a3b0b422e85b44`.
The Ubuntu 26.04 upgrade, reboot 2026-11-16, removed `/usr/bin/python3.12`.
Site-packages are untouched. The receipt is
`~/bench-scratch/venv-py312-repoint-RECEIPT-20261120.txt` on jetson-hub.

thor.local uses its system `/usr/bin/python3.12`, unchanged.

## Scope

On jetson-hub and on thor.local, every v3.1 campaign attempt from
attempt-001 onward uses the shim.

- jetson-hub, `~/bench-scratch/campaign-v3.1-4b9aa4d-jetsonhub`, attempt-001
  started 2026-11-20T13:53:07Z. There is no unshimmed v3.1 attempt on this
  host.
- thor.local, `~/bench-scratch/campaign-v3.1-4b9aa4d-thor`, attempt-001
  started 2026-11-20T14:00:51Z. attempt-001 was load-gated. Every v3.1
  attempt on this host uses the shim.

Any row collected without the shim is ungraded. The v3 attempts at
`777f0417` never used it. The recorded set is jetson-hub
`campaign-v3-777f041-jetsonhub`, attempt-001 through attempt-015.

Wall-time smokes that also used the shim, and that carry no energy grade:
jetson-hub `~/bench-scratch/thrml-smoke-20261120/cpu-shim` (the `777f0417`
binary) and
`~/bench-scratch/campaign-v3.1-4b9aa4d-jetsonhub/preflight-smoke/cpu-shim`
(v3.1); thor.local
`~/bench-scratch/campaign-v3.1-4b9aa4d-thor/preflight-smoke/cuda` and
`preflight-smoke/cpu` (v3.1). Each campaign directory's `LAUNCH.txt` records
the shim paths and their `exec` targets at arm time.

The comment `PREPARED, NOT IN USE until approved` inside the shim files is
out of date. It was left as written, so the hashes above do not change.

## Framework

framework-128gb-0000 runs no shim. The campaign directory is
`~/bench-scratch/campaign-v3.1-4b9aa4d-framework`. It uses `--scale
--quiet-gate` with the CPU-only venvs `thrml-venv` and
`thrml-main-2d96f4b4`, and it is labelled **CPU-suite-only AMD RAPL**. That
label supports no grid or GPU claim. This launch mode is the second approved
launch amendment.

## The code fix

The proper fix is a predeclared v3.2 change to `runner.py`. That change is
in progress on `harness/v3.2`.
