# Energy methodology v3.2

This is the v3.2 rule. It does not edit `bench/thrml-tsp/METHODOLOGY.md` or
`docs/METHODOLOGY-v3.1.md`. Those stay the frozen v3 text and the v3.1 text.
Brackets, the 3σ test, ABBA order, the load1 settle, the 0.05 parity line,
the stop rule, and the v3.1 rule that a trial-window refusal is stamped on
every meter row are unchanged. Their definitions are the v3 and v3.1 texts.

THRML A stays `thrml==0.1.4`. THRML B stays commit
`2d96f4b4b67bfa2625d89ed79f0a22c08af3b992`. JAX stays `0.12.0.1`.

## Base

`pre_rewrite_sha` is `777f0417340788428cf0525003277f9829ca6e3a`. v3.1's base
is `4b9aa4da02142efb58605b4c67cfa4a2191c1b16`. v3.2 does not regrade joules
collected under either tree.

## The JAX device is chosen before JAX starts

`runner.py` imported JAX and then asked `jax.devices()` which platform it
had. A CUDA build of JAX 0.12.0.1 registers the GPU at import and does not
register a CPU device unless `JAX_PLATFORMS` already names `cpu`. The CPU
suite then exited with `cpu was requested and JAX has no CPU device`. Live
v3.1 campaigns set `JAX_PLATFORMS` in a launch shim and exec'd the pinned
interpreter unchanged.

v3.2 sets `JAX_PLATFORMS` from the requested device before the JAX import.
`cpu` sets `cpu`. `cuda` sets `cuda`. `auto` still sets nothing. The runner
then records the platform and the device JAX actually bound.

Every energy receipt row carries `device_requested`, `jax_platform`, and
`jax_device`. On a ferrotherm row the three are null: ferrotherm is not a
JAX process. On a THRML row they are the request and the binding the runner
reported. A THRML row is refused when a binding was reported and it does not
match the request. `cpu` matches platform `cpu`. `cuda` matches platform
`gpu`. A CPU device is not a CUDA device. The refusal is on that row. It
does not change the trial-window thresholds and it does not change the 0.05
parity line.

## A THRML server builds only the workloads it will sample

`serve` builds and warms every workload in its spec before it prints
`READY`. That warmup stays where it is: before `READY`, and `READY` is
before any `COST`. `COST` is the metered window. v3.2 does not move the
warmup into that window, and it does not change THRML or JAX.

`--campaign` writes one `spec.txt` of 22 workloads: the 4 canonical, the
11 in `--scale`, and `grid_scale_ising` at sides 64, 128, 256, 512, 1024,
2048 and 4096. Side 4096 is 16,777,216 sites. The suite phase was started
with that file, so the CPU-suite server built the 2048 and 4096 grids
before `READY` and was killed at about 12.5 GB. The same server given only
the 15-workload suite spec sits at about 1.3 to 1.5 GB. This is separate
from the `--skip-thrml` audit growth below. That growth runs with no THRML
process at all.

v3.2 hands each server a spec of the workloads that server will sample.
The suite server gets the suite workloads. Each grid server gets one side,
the spec written for that side before the process is armed. A server does
not construct a grid it will not be asked to sample.

## A repetition is one execution of the sampler

The metered window is still warmup plus draws, repeated until the window has
lasted 2 s, and joules are still divided by that repetition count. A call
that does not execute the sampler is not a repetition, and it does not keep
the window open. v3.1 counted those calls. Two paths did it.

With `--skip-thrml` the THRML side has no process. The loop still ran until
2 s had passed. On an x86_64 host that is tens of millions of empty
iterations per side. The phase then kept one outside-step record per
iteration. Measured on an x86_64 VM with a readable RAPL package counter, RSS rose by
about 1.4 GB per skipped side and was still rising at 6.1 GB. The same
command with no counter returns before that loop. jetson-hub is not a v3.2
campaign host: that machine does not have enough memory for these campaigns.
The growth above was reproduced on the VM.

The records are appended after the post-trial bracket, so that allocation
is outside the counter window. The empty loop that produced the count ran
inside the window. A skipped THRML row that was metered is the joules of
that loop.

With THRML running, non-grid `COST` called `sample_states` on an empty
block. In THRML 0.1.4, `Block.node_type` raises `ValueError: Block is empty
and doesn't have a node type` before a sweep. Measured with that pin and
JAX 0.12.0.1, one `serve` process, `COST` of `thrml_readme_ising_chain_n5`
failed for all 39 seeds in the sweep: 0 through 32, 100, 101,
2147483647, 260801615, 260801616, and 260811615. Seed 1 is in that list.
`ESS` and `RUN` at seed 1 both returned a chain. The seed is not the
input that selects the error. The observer block
is `Block([])` on every non-grid `COST`. `RUN` and `ESS` pass the real
even/odd blocks and do not raise, including at seed 1. `serve` catches the
exception and prints `ERR`, so the process stays up. A log that shows the
error on some lines of that workload and not others is `COST` versus
`RUN`/`ESS`, not a seed that sometimes works. The same 2 s loop counted
the failures. Grid `COST` already used a null observer and did run the sweeps.

v3.2 counts an execution only when ferrotherm's warmup-plus-draws returns,
or when THRML `COST` returns a positive sweep count. Non-grid `COST` uses
the same null observer as grid `COST`: the sweep runs, and states are not
stacked. There is no RSS cap and no watchdog.

## What a finished receipt can contain

A receipt is written at the end of the phase. The audit allocation grows
until the process is killed, and a killed run writes no receipt. The
allocation itself sits after the bracket, so it is not inside a counter
window that was already closed.

A finished receipt can still hold biased joules. Non-grid THRML `COST` on
v3.1 and on `777f0417` metered the exception loop, not Gibbs sweeps. A
`--skip-thrml` row that was metered includes the empty loop's CPU. Grid
THRML `COST` did run sweeps; that window is the sampler. A suite server
that was killed while building the campaign spec wrote no receipt for
that phase. v3.2 does not rewrite those receipts and does not regrade
them.

## Receipts

A receipt written by the v3.2 harness carries `rules_version` `v3.2`.
