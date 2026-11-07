"""Which JAX platform a THRML run is allowed to use.

`cuda` means a JAX device whose platform is `gpu`. A CPU device is never
reported as CUDA. `auto` prefers a GPU when JAX has one.
"""


def choose_platform(requested: str, platforms: list[str]) -> str:
    have = set(platforms)
    if requested == "cpu":
        if "cpu" not in have:
            raise ValueError("cpu was requested and JAX has no CPU device")
        return "cpu"
    if requested == "cuda":
        if "gpu" not in have:
            raise ValueError(
                "cuda was requested and JAX has no GPU device. A CPU device is not a CUDA device."
            )
        return "gpu"
    if requested != "auto":
        raise ValueError(f"device must be auto, cpu, or cuda, got {requested}")
    if "gpu" in have:
        return "gpu"
    if "cpu" in have:
        return "cpu"
    raise ValueError("JAX reported no devices")
