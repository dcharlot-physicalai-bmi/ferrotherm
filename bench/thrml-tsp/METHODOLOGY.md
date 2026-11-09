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
