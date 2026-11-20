"""Device choice, without importing JAX."""

from pathlib import Path

from platform_choice import (
    choose_platform,
    jax_platforms_value,
    pin_jax_platform,
    requested_device,
)
from sweep_count import executed_gibbs_sweeps, schedule_sweeps


def main() -> None:
    assert choose_platform("auto", ["cpu"]) == "cpu"
    assert choose_platform("auto", ["cpu", "gpu"]) == "gpu"
    assert choose_platform("cpu", ["cpu", "gpu"]) == "cpu"
    assert choose_platform("cuda", ["gpu"]) == "gpu"
    try:
        choose_platform("cuda", ["cpu"])
    except ValueError as err:
        text = str(err)
        if "CUDA" not in text and "cuda" not in text:
            raise SystemExit(f"the refusal does not name cuda: {text}")
    else:
        raise SystemExit("a CPU-only JAX was accepted as cuda")
    if requested_device(["--spec", "s", "--device", "cpu"]) != "cpu":
        raise SystemExit("argv --device cpu was not read")
    if requested_device(["--device=cuda", "--serve"]) != "cuda":
        raise SystemExit("argv --device=cuda was not read")
    if requested_device(["--spec", "s"]) != "auto":
        raise SystemExit("a missing --device is not auto")
    if jax_platforms_value("cpu") != "cpu" or jax_platforms_value("cuda") != "cuda":
        raise SystemExit("cpu and cuda must pin JAX_PLATFORMS to that token")
    if jax_platforms_value("auto") is not None:
        raise SystemExit("auto must not pin JAX_PLATFORMS")
    env: dict[str, str] = {}
    pin_jax_platform("cpu", env)
    if env.get("JAX_PLATFORMS") != "cpu":
        raise SystemExit(f"pin did not set cpu, got {env}")
    pin_jax_platform("auto", env)
    if env.get("JAX_PLATFORMS") != "cpu":
        raise SystemExit("auto overwrote a platform that was already selected")
    pin_jax_platform("cuda", env)
    if env.get("JAX_PLATFORMS") != "cuda":
        raise SystemExit("cuda did not replace the cpu pin")
    runner = Path(__file__).with_name("runner.py").read_text(encoding="utf-8")
    pin_at = runner.index("pin_jax_platform(")
    jax_at = runner.index("\nimport jax\n")
    if pin_at > jax_at:
        raise SystemExit("runner.py imports JAX before pinning the platform")
    cost = runner.split("def _state_cost", 1)[1].split("\ndef ", 1)[0]
    if "Block([])" in cost or "_NULL_OBSERVER" not in cost:
        raise SystemExit("non-grid COST still stacks states or uses an empty block")
    if "_meter_sweeps" not in cost or "block_until_ready" not in cost:
        raise SystemExit("non-grid COST does not report executed sweeps after the call returns")
    grid = runner.split("def _grid_cost", 1)[1].split("\ndef ", 1)[0]
    if "_meter_sweeps" not in grid:
        raise SystemExit("grid COST does not report executed sweeps")
    # A schedule that does not enter _run_blocks executed nothing.
    if executed_gibbs_sweeps(0, 1, 1) != 0 or executed_gibbs_sweeps(0, 0, 1) != 0:
        raise SystemExit("a non-positive warmup with at most one sample is not 0 sweeps")
    if executed_gibbs_sweeps(0, 4, 0) != 0:
        raise SystemExit("a non-positive step invented sweeps")
    # draws >= 1 matches warmup + draws * thin, which is what collect runs.
    if schedule_sweeps(400, 2000, 1) != 2400:
        raise SystemExit(f"pinned chain sweeps were {schedule_sweeps(400, 2000, 1)}, want 2400")
    if schedule_sweeps(400, 8, 1) != 408:
        raise SystemExit("warmup 400, draws 8, thin 1 is not 408 executed sweeps")
    if executed_gibbs_sweeps(5, 1, 3) != 5:
        raise SystemExit("one sample runs only the warmup")
    print("ok")


if __name__ == "__main__":
    main()
