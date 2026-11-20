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

## A repetition is one execution of the sampler

The metered window is still warmup plus draws, repeated until the window has
lasted 2 s, and joules are still divided by that repetition count. A call
that does not execute the sampler is not a repetition, and it does not keep
the window open. v3.1 counted those calls. Two paths did it.

With `--skip-thrml` the THRML side has no process. The loop still ran until
2 s had passed. On an x86_64 host that is tens of millions of empty
iterations per side. The phase then kept one outside-step record per
iteration. Measured here with a readable RAPL package counter, RSS rose by
about 1.4 GB per skipped side and was still rising at 6.1 GB. The same
command with no counter returns before that loop. A Jetson run that finishes
in about 29 s at about 20 MB is the quiet-host wait (15 attempts, 2 s
apart) leaving the phase before the loop, not a different sampler.

The records are appended after the post-trial bracket, so that allocation
is outside the counter window. The empty loop that produced the count ran
inside the window. A skipped THRML row that was metered is the joules of
that loop.

With THRML running, non-grid `COST` called `sample_states` on an empty
block. In THRML 0.1.4, `Block.node_type` raises `ValueError` on an empty
block, so the call failed before a sweep. The same 2 s loop counted the
failures, and the harness then sent that many `ESS` requests. Each `ESS`
collects states. That is the THRML server's growth. THRML and JAX are not
modified. Grid `COST` already used a null observer and did run the sweeps.

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
THRML `COST` did run sweeps; that window is the sampler. v3.2 does not
rewrite those receipts and does not regrade them.

## Receipts

A receipt written by the v3.2 harness carries `rules_version` `v3.2`.
