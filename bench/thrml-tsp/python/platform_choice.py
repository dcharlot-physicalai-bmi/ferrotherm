"""Which JAX platform a THRML run is allowed to use.

`cuda` means a JAX device whose platform is `gpu`. A CPU device is never
reported as CUDA. `auto` prefers a GPU when JAX has one.
"""


def requested_device(argv: list[str]) -> str:
    """The `--device` value on a runner command line. Default `auto`.

    Read this before JAX is imported. `bind_device` runs too late: a CUDA
    build of JAX has already dropped the CPU device by then.
    """
    device = "auto"
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--device" and i + 1 < len(argv):
            device = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--device="):
            device = arg.split("=", 1)[1]
            i += 1
            continue
        i += 1
    return device


def jax_platforms_value(requested: str) -> str | None:
    """`JAX_PLATFORMS` for this request, or `None` when the env must be left alone.

    `cuda` is the token JAX 0.12 takes for the CUDA platform. The device list
    then reports that platform as `gpu`. `auto` does not pin a platform.
    """
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda"
    return None


def pin_jax_platform(requested: str, env: dict[str, str]) -> None:
    """Select the device before JAX initialises. A later `jax.devices()` cannot."""
    value = jax_platforms_value(requested)
    if value is None:
        return
    env["JAX_PLATFORMS"] = value


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
