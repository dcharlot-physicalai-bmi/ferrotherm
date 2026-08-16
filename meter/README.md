# ferrotherm-meter

Joules **measured on the machine that ran the work**, instead of borrowed from another vendor's
datasheet.

[ferrotherm](https://github.com/dcharlot-physicalai-bmi/ferrotherm)'s core ledger prices a run
against a `Prices` device model — useful, and explicitly a projection. This crate is the other half:
it turns a run's operation counts into per-operation energy from **wall power you actually
observed**.

```sh
cargo add ferrotherm-meter
```

## Idle is subtracted, and both numbers are shown

A machine drawing 20 W doing nothing must not charge that to your workload. A "measurement" that
forgets to subtract it reports mostly the cost of the computer being switched on. So the baseline
and the delta are both reported — the subtraction is visible rather than assumed.

## What it refuses to answer

This is the part worth reading, because the failure mode of energy measurement is a confident
number, not an error.

- **A run too short to measure.** Backends report power on a fixed tick, so a 50 ms run sampled
  every 200 ms collects zero or one sample. One sample of a fluctuating quantity is not an estimate
  of its mean, so it returns an error naming the shortfall instead of a number computed from too
  little.
- **A run that does not rise above idle.** If the delta sits inside the noise, there is no signal to
  attribute, and the message says to run a bigger model rather than more repeats — repeats do not
  help when the effect is smaller than the wander.
- **A delta below the 3σ noise floor**, where σ is the baseline's own standard deviation over its
  samples. An earlier version applied the floor with a `max(0.0)` that silently turned a −0.05 W
  delta into zero joules; the floor now applies to every delta regardless of sign.
- **A mixed workload**, where no single operation dominates. It attributes energy to the operation
  that dominated and refuses when none did, because splitting a joule between operations by
  assumption is how a per-operation figure becomes fiction.

## Backends

- **macOS** — `macmon pipe`, reading the SoC's own power counters. Detected on `PATH`.
- **Jetson / Linux** — the INA3221 rails under `/sys/bus/i2c/.../in_power*_input` are the
  equivalent. **Not implemented.** The Jetson on our tailnet has been offline, and a backend nobody
  can run is a backend nobody has tested. The trait is the seam it slots into.

Apache-2.0. From the Institute for Physical AI @ BMI.
