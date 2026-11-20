"""Block-Gibbs iterations one THRML metered call executes.

`sample_with_observation` runs `_run_blocks` once for the warmup, and then
once per sample after the first. `_run_blocks` returns immediately when its
count is not positive, so a non-positive warmup or step adds nothing. A
schedule with one sample or none runs only the warmup. The harness adds
these reports. It does not multiply them by a repetition count.
"""


def executed_gibbs_sweeps(n_warmup: int, n_samples: int, steps_per_sample: int) -> int:
    """Iterations the submitted schedule runs. Zero is a real report."""
    warm = n_warmup if n_warmup > 0 else 0
    if n_samples <= 1:
        return warm
    step = steps_per_sample if steps_per_sample > 0 else 0
    return warm + (n_samples - 1) * step


def schedule_sweeps(warmup: int, draws: int, thin: int) -> int:
    """Sweeps for the schedule COST submits.

    `n_warmup` is `warmup + thin` so the first kept state matches ferrotherm's
    collect, which does not keep the state at the end of burn-in. `thin` is
    at least 1. For `draws >= 1` this equals `warmup + draws * thin`.
    """
    step = thin if thin > 0 else 1
    return executed_gibbs_sweeps(warmup + step, draws, step)
