"""Device choice, without importing JAX."""

from platform_choice import choose_platform


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
    print("ok")


if __name__ == "__main__":
    main()
