# Energy methodology

This is the rule the harness applies, and the fields a receipt has to carry.
Joules come from a hardware counter or they are absent. A process id attributes
or refuses. It is never subtracted from joules.

Grades are per phase and per meter. Package, core, `soc-telemetry`, `vin`, and
each `subrail:` scope stay in their own tables. Two hosts are not pooled.

## Phases

`phases.preflight`, `phases.energy`, and `phases.accuracy` each have their own
grade. The bundle `evidence_grade` is the energy-phase grade. It is `quiet`
only when the energy phase itself is quiet. `contended host` stays contended.
Anything else on the energy phase is `load_gated`. A preflight that held does
not promote the bundle.

When `--quiet-gate` holds, energy runs immediately and accuracy follows it.
THRML is not imported during that gap. The THRML process, when the B arm needs
it, is started inside the first THRML trial window. Its CPU is the arm being
measured.

## Idle brackets

`load1` is the 1-minute load average. The quiet threshold is 2 and is not
changed.

Before the baseline and before each pair's brackets, the harness waits until
`load1` has stayed at or under 2 for 60 consecutive samples at 1 Hz. The wait
is bounded at 600 samples. The samples are `energy.baseline_settle_load1` for
the baseline and `settle_load1` on each trial. If the streak never completes,
the energy phase is `load_gated` and the receipt says why.

After that wait, every 1 Hz sample through the bracket has to be finite and at
or under 2. An empty series is a refusal. The first sample that fails is
recorded and the phase stops.

## Trial windows

`load1` is not a trial-window criterion. It counts runnable threads, so the
multithreaded workload under test, including THRML and JAX, raises it. That is
what a post-trial bracket sees. Refusing the trial window on `load1` refuses
the measurement for doing the work. Brackets use `load1`. Trial windows do not.

The window is sampled at 1 Hz. `window_load1` and `ps_attribution` are
evidence. `ps_attribution` lists processes that are not this harness. Neither
field changes joules.

On Linux the refusal signal is the mean busy fraction of cores outside
`--cpus`, from `/proc/stat`. The receipt stamps `other_core_mean_max` and
`other_core_sample_max`. The defaults are 0.05 and 0.15. A trial is refused
when the mean is above the first or any sample is above the second, or when
the window recorded no finite sample. A quiet energy grade on Linux requires
`--cpus`. Without a pin the outside-core fraction is undefined.

On macOS two checks apply. The pre and post idle brackets have to agree within
3 sigma, the same drift test the bracket already computes. Separately, each
1 Hz sample sums the CPU time gained by processes that are not this harness
and not its children, and divides by the wall time of that second. That is a
fraction of one core. The defaults are a mean at most 0.15 and no sample above
0.50, stamped as `foreign_cpu_mean_max` and `foreign_cpu_sample_max`. A
failing sample refuses the trial. The fraction is not subtracted from joules.

`--other-core-mean`, `--other-core-max`, `--foreign-cpu-mean`, and
`--foreign-cpu-max` override the defaults. The values that were applied are
`energy.trial_window_rule`, including `enforce` and
`why_load1_is_not_a_trial_criterion`.

## Host lock

Another measurement on the same machine is foreign load. The harness takes
`flock(2)` `LOCK_EX` on `/tmp/bmi-energy.lock` before preflight and holds it
through the energy phase, the post-trial idle bracket, and the accuracy phase.
`--lock` changes the path. The constants are the same on Linux and macOS:
`LOCK_EX` is 2, `LOCK_NB` is 4, `LOCK_UN` is 8.

If the lock is busy the harness polls `LOCK_EX|LOCK_NB`, logs the path, the
holder pid read from the file, and the seconds already waited, and sleeps.
The wait is bounded by `--lock-wait-s`, default 3600. The run does not proceed
unlocked. If the wait ends without the lock, the energy grade is refused with
reason `host lock not held` and neither phase runs.

`host_lock` on the receipt has `path`, `acquired_utc`, `released_utc`,
`wait_s`, `holder_pid`, `held`, and `reason`. The timestamps are UTC
`YYYY-MM-DDThh:mm:ss.mmmZ`. `reason` is null when the lock was held.

## Receipt

`schema` is `ferrotherm-thrml-bench/1`.

| Field | Where it lives |
| --- | --- |
| Bundle grade | `evidence_grade` |
| Per-phase grades | `phases.preflight.grade`, `phases.energy.grade`, `phases.accuracy.grade` |
| Host lock | `host_lock.path`, `acquired_utc`, `released_utc`, `wait_s`, `holder_pid`, `held`, `reason` |
| Trial-window rule | `energy.trial_window_rule` |
| Baseline settle | `energy.baseline_settle_load1` |
| Per-trial settle | `energy.trials[].settle_load1` |
| Trial-window load, evidence only | `energy.trials[].window_load1` |
| Trial-window refusal signal | `energy.trials[].window_signal` |
| Process attribution | `energy.trials[].ps_attribution` |
| Trial refusal | `energy.trials[].trial_refusal` |

Meters stay on their own scope inside `energy.summaries` and `energy.trials`.
A missing counter is an absent joule, not a zero.

## Large-grid sweep

`--grid-scale` replaces the pinned workload list with open Ising grids, `J = 1`,
`beta = 0.3`, sides 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096.
`--sides` may name a subset of that list. A side not in the list is an error.
It does not drop a side that was requested and then failed.

The host lock, the quiet gate, the idle brackets, and the trial-window rule are
the ones above. They are not relaxed for a large grid.

Each side is its own workload. Each meter is its own summary. Package, core,
SoC telemetry, `vin`, and `subrail:` scopes are not pooled, and two hosts are
not pooled. Baseline A and baseline B are not pooled with each other, and the
ferrotherm rows paired with A are not pooled with the ferrotherm rows paired
with B.

### What is measured, separately

For a grid-scale trial the joule window runs sweeps only. It does not score
the trace. After the post-trial idle bracket, the same seeds are run again and
the energy trace is scored. That second run is not inside the counter window.

| Field | Definition |
| --- | --- |
| `seconds_per_sweep` | per-repetition window time divided by `warmup + draws * thin` |
| `joules_per_sweep` | attributable incremental joules divided by that same sweep count |
| `ess_per_sweep` | `ess / (draws * thin)` from the trace measured outside the window |
| `joules_per_ess` | attributable incremental joules divided by that `ess`, with `uncertainty_j_per_ess` |

`joules_per_ess` is not `joules_per_sweep / ess_per_sweep`. The joule window
includes warmup sweeps. `ess_per_sweep` counts only the sweeps between kept
draws. Both numbers are on the receipt. The uncertainty budget is the existing
one: Student-t half-width of the trial `joules_per_ess` values, plus the mean
quantization, cadence-edge, and idle-bracket terms from `combine_uncertainty`.

### Parity

Each sampler is judged on its own. A is not B's reference, and B is not A's.

Sides 8 and 16 have a transfer-matrix mean. The chain mean has to be strictly
inside 0.05 per site of it. Larger sides have no finite-lattice mean on the
receipt. The per-site gap to Onsager has to be strictly smaller than the exact
16×16 gap, and two seeds have to agree to strictly less than 0.05 per site.
Those are the gates in `oracle`'s Ising tests. A size that fails is
`parity.ok = false` with the numbers in `reason`. The process exits non-zero.
The other sampler's row is left as it measured.

### Profile

`--profile` requires `--grid-scale`. It takes the host lock and runs no energy
phase. `profile.json` records, per side, elapsed `wall_ns` and the CPU time in
reading spins (`memory_ns`), forming the field or the threshold (`field_ns`),
and drawing accept bits (`rng_ns`). With more than one thread the three CPU
times are summed across threads. They are not joules.

A wall-time profile on a 4-vCPU cloud VM does not say where the curve bends on
an i9-13900H or on Thor. Those hosts have different core counts, caches, and
memory systems. The bend is a measurement on the host that will be quoted, with
that host's own `--cpus` list. The VM numbers are a check that the levers run
and that parity holds. They are not that curve.

A grid-scale energy run requires `--cpus`. There is no default list. The host
driver passes the cores it is measuring. `--profile` does not measure energy
and does not require the flag.

### Levers

`--lever` may be repeated. The names are `parallel`, `simd`, `tiled`,
`sparse-rng`, `incremental`, `combined`, and `gpu`. `gpu` does not combine
with a CPU lever. Unknown names are an error. All off is the scalar
checkerboard, one `f64` draw per site.

`parallel` and `tiled` do not change the draws. `simd` packs 64 sites of one
colour into a word and refuses an odd side instead of falling back. `sparse-rng`
uses a nine-entry threshold table of `kernel::p_up` and compares from the high
bit, stopping when every site has resolved. The table differs from `p_up` by
at most `2^-32`. The leading bits are the high bits of the generator. The low
bits of nearby streams are serially correlated, and using them as the leading
bit biased a 2×2 chain off its Boltzmann law.

`incremental` keeps an i8 neighbour sum per site, or, with `simd`, replaces the
trit adder with a popcount of the four neighbour bit planes. The field value is
the same number the trit adder produced. The scalar i8 path draws with
`next_u32() < threshold`. Its 2×2, 3×3 and 4×4 laws match exact enumeration.
`combined` is `parallel`, `simd`, `tiled`, and `incremental` together. It is
not a mix with `sparse-rng`.

### Side-8 two-seed sample

The 0.05 per-site line is unchanged. On an 8×8 it is an absolute gap of 3.2 in
the mean energy. Warmup 64 and 256 draws, on seeds `seed0` and `seed0+1`,
makes incremental print exactly −36.375 and −39.578125 (0.050048828125 per
site) on every host, because the sampler is deterministic. Scalar on that same
short trace is 0.040 per site and simd is 0.025. A 24-pair pilot on seeds
`seed0+10000` upward, warmup 400, all three levers, had a worst gap of 0.0527
at 1024 draws, 0.0217 at 2048, and 0.0133 at 4096. Side 8 therefore uses warmup
400 and 4096 draws, the smallest power of two whose pilot worst gap is below
0.02. A shorter `--draws` or `--warmup` does not shrink side 8. The same count
is used for every lever. The pilot seeds are not the gate seeds. `parity_design`
on the receipt records the rule and the pilot gaps.

### Receipt hashes

`spec_sha256` is SHA-256 of `spec.txt`. `receipt_sha256` is SHA-256 of that
receipt file with the `receipt_sha256` field set to 64 zeros. Replacing the
published hash with those zeros and hashing the file again reproduces it.
`report.json`, `profile.json`, and `quiet-gate.json` each carry both.

`gpu` is a packed checkerboard compute shader: one colour's spins in `u32`
words, one thread per word, and `grid::packed_hash` as the draw.
The CPU mirror is `sweep_packed_cpu`. The bench has to be built with
`--features gpu`. No adapter is an error, not a CPU run. A software rasteriser
is labelled not hardware. The older CSR gather shader is not this lever.

### ESS per sweep and ESS per second

`mixing.ess_per_sweep` is `ess / (draws * thin)`. `mixing.ess_per_second` is
that rate divided by wall seconds. `mixing.wall_ns_per_sweep` is chain seconds
times 1e9 divided by `warmup + draws * thin`. Those seconds exclude
`jit_seconds`. The joule window's `seconds_per_sweep` is a different timer:
on a THRML row, `wall_ns_per_sweep` is the DONE line, which starts after READY.
They are separate because a faster sweep and a better-mixing sweep are
different claims. A higher THRML `ess_per_sweep` at a large side means its
blocked sampler mixed better per sweep. The ratio is null when either side
has no finite ESS. A null ratio is not a ferrotherm win.

### THRML's own memory limit

Grid-scale accuracy runs one THRML process per side, and so does the energy
pairing. The label does not depend on `--device`: CPU and CUDA use the same
exit check. A SIGKILL (signal 9, or exit code 137) is `THRML_LIMIT oom`. So is
exit code 1 whose stderr names an allocator failure (`RESOURCE_EXHAUSTED`,
`out of memory`, `CUDA_ERROR_OUT_OF_MEMORY`). The ratio is null. Sides that
finished stay in the receipt. The kill is THRML's memory limit. It is not a
ferrotherm result and it is not a win. A Python `MemoryError` or a JAX
allocation error prints a `LIMIT` line and the runner continues.

`parallel` offers whatever `--threads` says, then caps it at 4 below side 512.
Eight threads beat four only from side 512 up on the host that compared them.

The packed GPU chain is one submission. Each colour of one sweep is one
dispatch, because a workgroup cannot barrier against another workgroup, so a
sweep cannot be a single dispatch. `sweeps_per_submit` is the whole chain.
`setup_ns` is pipeline and buffer creation and sits inside the trial's wall
time. The bond sum is read once, not the spin array after every sweep.

### Two THRML baselines

A is `thrml==0.1.4`, `--python`. B is commit `2d96f4b4b67bfa2625d89ed79f0a22c08af3b992`
(2026-11-03), `--python-b` or `THRML_PYTHON_B`, its own venv. The runner checks
the shape of `IsingSamplingProgram` and, for B, `direct_url.json`'s commit. The
wrong tree is an error. It is not relabelled.

B is six commits past v0.1.4. The cost-relevant ones are #77 (`2d54cd84`,
pluggable Ising backends; `IsingSamplingProgram` is a logical program), #78
(`9cfe3303`, `BlockGibbsSpec` superblock tuples), #74 (`c6948767`, the
categorical `inf*0` fix), and `5911729f` (Ising KL). B is re-measured. It is
not assumed to match A.

Receipt fields `thrml` and `thrml_main_2d96f4b4` are the two accuracy blocks.
Energy rows carry `pairing` `A` or `B` and `fabric`:

| Fabric | What ran |
| --- | --- |
| `ferrotherm-cpu` | the CPU levers |
| `ferrotherm-gpu` | wgpu |
| `thrml-0.1.4-jax-cpu` | A, `--device cpu` |
| `thrml-0.1.4-jax-cuda` | A, `--device cuda` |
| `thrml-main-2d96f4b4-jax-cpu` | B, `--device cpu` |
| `thrml-main-2d96f4b4-jax-cuda` | B, `--device cuda` |

`--device cuda` is a JAX platform `gpu`. A CPU device is not reported as CUDA.
`--device auto` is labelled `jax-cpu` in the fabric field only when the flag
was `cpu`. Operators running the sweep pass `cpu` or `cuda` explicitly. The
runner's `META` line is the platform JAX actually bound.

### Commands for the host drivers

A, CPU, full side list, quiet gate, pinned cores. The venv is the pinned 0.1.4
one. Repeat with `--device cuda` only in a venv whose JAX is `jax[cuda13]`.

```sh
cargo build --release -p ferrotherm-thrml-bench
./target/release/ferrotherm-thrml-bench \
  --out results/grid-scale-a-cpu \
  --grid-scale --quiet-gate --cpus CORES \
  --device cpu \
  --python /path/to/thrml-0.1.4-venv/bin/python \
  --python-b /path/to/thrml-main-2d96f4b4-venv/bin/python
```

B uses the same command. Both baselines are in that run. They are not a second
invocation with A's numbers copied across.

CUDA, on the machine that has `jax[cuda13]` (thor.local keeps one under
`~/bench-scratch/thrml-venv-cuda`). Install B into a different venv that has
the same CUDA JAX. Do not point `--python-b` at A's tree.

```sh
./target/release/ferrotherm-thrml-bench \
  --out results/grid-scale-cuda \
  --grid-scale --quiet-gate --cpus CORES \
  --device cuda \
  --python ~/bench-scratch/thrml-venv-cuda/bin/python \
  --python-b ~/bench-scratch/thrml-main-2d96f4b4-cuda/bin/python
```

Ferrotherm GPU lever, built with wgpu, still against both THRML baselines:

```sh
cargo build --release -p ferrotherm-thrml-bench --features gpu
./target/release/ferrotherm-thrml-bench \
  --out results/grid-scale-ferrotherm-gpu \
  --grid-scale --lever gpu --quiet-gate --cpus CORES \
  --device cpu \
  --python /path/to/thrml-0.1.4-venv/bin/python \
  --python-b /path/to/thrml-main-2d96f4b4-venv/bin/python
```

One CPU lever at a time, so the effect is not confounded. `LEVER` is
`parallel`, `simd`, `tiled`, `sparse-rng`, `incremental`, or `combined`.

```sh
./target/release/ferrotherm-thrml-bench \
  --out results/grid-scale-lever-$LEVER \
  --grid-scale --lever $LEVER \
  --quiet-gate --cpus CORES --device cpu \
  --python /path/to/thrml-0.1.4-venv/bin/python \
  --python-b /path/to/thrml-main-2d96f4b4-venv/bin/python
```

Profile, no joules, same levers:

```sh
./target/release/ferrotherm-thrml-bench \
  --out results/grid-scale-profile-$LEVER \
  --grid-scale --profile --lever $LEVER --threads N \
  --device cpu \
  --python /path/to/thrml-0.1.4-venv/bin/python \
  --python-b /path/to/thrml-main-2d96f4b4-venv/bin/python
```

Install B beside A:

```sh
bash bench/thrml-tsp/python/install-baseline-b.sh /path/to/thrml-main-2d96f4b4-venv
```
