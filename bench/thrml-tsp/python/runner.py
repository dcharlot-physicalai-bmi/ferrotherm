"""Pinned THRML 0.1.4 runner.

Reads the spec text ferrotherm-thrml-bench writes and samples each workload with
block Gibbs. States come back as the modeller's values: spins in {+1, -1}, Potts
labels in 0 .. q-1. This process does not compute total variation, ESS, or joules.
The Rust harness scores both chains with one estimator and one energy probe.

Schedule alignment with ferrotherm's Sampler::collect: that call discards `warmup`
sweeps, then keeps a state every `thin` sweeps, and does not keep the state sitting
at the end of the burn-in. THRML's SamplingSchedule keeps the state at the end of
`n_warmup` as sample 0. Setting n_warmup = warmup + thin and steps_per_sample = thin
makes the first kept state and the spacing match.

No power tool is invoked here. Wall and GPU counters are read by the Rust process
around this one, so both runners are on the same instrument.
"""

from __future__ import annotations

import argparse
import sys
import time

import jax
import jax.numpy as jnp

import thrml
from platform_choice import choose_platform
from thrml import Block, CategoricalNode, SamplingSchedule, SpinNode, sample_states
from thrml.block_sampling import BlockGibbsSpec
from thrml.factor import FactorSamplingProgram
from thrml.models.discrete_ebm import (
    CategoricalEBMFactor,
    CategoricalGibbsConditional,
    SpinEBMFactor,
    SpinGibbsConditional,
)
from thrml.models.ising import IsingEBM, IsingSamplingProgram


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()
    try:
        platform, device = bind_device(args.device)
    except ValueError as err:
        raise SystemExit(str(err)) from err
    workloads = parse_spec(open(args.spec, encoding="utf-8").read())
    if args.serve:
        return serve(workloads, device, platform)
    if not args.out:
        raise SystemExit("--out is required unless --serve")
    lines = [
        "META thrml={thrml} jax={jaxv} jaxlib={jaxlib} equinox={eqx} device={dev} jax_platform={plat}".format(
            thrml=thrml.__version__,
            jaxv=jax.__version__,
            jaxlib=_jaxlib_version(),
            eqx=_eqx_version(),
            dev=str(device),
            plat=platform,
        )
    ]
    with jax.default_device(device):
        for w in workloads:
            lines.extend(run_workload(w))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return 0


def _jaxlib_version() -> str:
    import importlib.metadata

    return importlib.metadata.version("jaxlib")


def _eqx_version() -> str:
    import importlib.metadata

    return importlib.metadata.version("equinox")


def parse_spec(text: str) -> list[dict]:
    out = []
    cur = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        head, _, rest = line.partition(" ")
        if head == "WORK":
            cur = {"fields": dict(part.split("=", 1) for part in rest.split())}
        elif head == "SOURCE":
            cur["source"] = rest
        elif head == "BIASES":
            cur["biases"] = [float(x) for x in rest.split()] if rest else []
        elif head == "EDGES":
            cur["edges"] = [_triple(tok, 3) for tok in rest.split()] if rest else []
        elif head == "TRIPLES":
            cur["triples"] = [_triple(tok, 4) for tok in rest.split()] if rest else []
        elif head == "BLOCKS":
            cur["blocks"] = []
            for tok in rest.split():
                if tok == "-":
                    cur["blocks"].append([])
                else:
                    cur["blocks"].append([int(x) for x in tok.split(",")])
        elif head == "END":
            out.append(_finish(cur))
            cur = None
        else:
            raise SystemExit(f"unrecognised spec line: {line}")
    if cur is not None:
        raise SystemExit("spec ended inside a workload")
    return out


def _triple(tok: str, n: int):
    parts = tok.split(",")
    if len(parts) != n:
        raise SystemExit(f"expected {n} comma-separated fields, got {tok}")
    nums = [float(p) for p in parts]
    return nums


def _finish(cur: dict) -> dict:
    f = cur["fields"]
    return {
        "id": f["id"],
        "kind": f["kind"],
        "beta": float(f["beta"]),
        "n": int(f["n"]),
        "q": int(f["q"]),
        "warmup": int(f["warmup"]),
        "draws": int(f["draws"]),
        "thin": int(f["thin"]),
        "trials": int(f["trials"]),
        "seed0": int(f["seed0"]),
        "source": cur.get("source", ""),
        "biases": cur["biases"],
        "edges": cur["edges"],
        "triples": cur["triples"],
        "blocks": cur["blocks"],
    }


def bind_device(requested: str):
    """Pick a JAX device. `cuda` requires platform `gpu` and does not fall back to CPU."""
    devices = list(jax.devices())
    platform = choose_platform(requested, [d.platform for d in devices])
    device = next(d for d in devices if d.platform == platform)
    return platform, device


def serve(workloads: list[dict], device, platform: str) -> int:
    """JIT once, then sample one seed per `RUN id seed` line. QUIT ends the process."""
    ready = {}
    with jax.default_device(device):
        for w in workloads:
            program, init, observe, decode = _build(w)
            key = jax.random.key(0)
            burned = sample_states(key, program, SamplingSchedule(1, 2, 1), init, [], observe)
            jax.block_until_ready(burned)
            ready[w["id"]] = (w, program, init, observe, decode)
        print(f"READY platform={platform}", flush=True)
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            if line == "QUIT":
                return 0
            parts = line.split()
            if len(parts) != 3 or parts[0] != "RUN":
                print(f"ERR unrecognised {line}", flush=True)
                continue
            wid, seed_s = parts[1], parts[2]
            if wid not in ready:
                print(f"ERR unknown workload {wid}", flush=True)
                continue
            w, program, init, observe, decode = ready[wid]
            seed = int(seed_s)
            lines = _one_trial(w, program, init, observe, decode, seed, 0.0)
            print("\n".join(lines), flush=True)
    return 0


def _build(w: dict):
    kind = w["kind"]
    if kind == "ising":
        return _ising_program(w)
    if kind == "potts":
        return _potts_program(w)
    if kind == "hubo":
        return _hubo_program(w)
    raise SystemExit(f"unknown kind {kind}")


def _one_trial(w, program, init, observe, decode, seed: int, jit_s: float) -> list[str]:
    thin = max(1, w["thin"])
    schedule = SamplingSchedule(w["warmup"] + thin, w["draws"], thin)
    key = jax.random.key(seed)
    t0 = time.perf_counter()
    samples = sample_states(key, program, schedule, init, [], observe)
    jax.block_until_ready(samples)
    seconds = time.perf_counter() - t0
    states = decode(samples)
    if len(states) != w["draws"] or len(states[0]) != w["n"]:
        raise SystemExit(f"{w['id']} seed {seed}: shape {len(states)} x {len(states[0]) if states else 0}")
    lines = [
        f"WORK id={w['id']} seed={seed} kind={w['kind']} seconds={seconds:.9f} jit_seconds={jit_s:.9f}"
    ]
    for row in states:
        lines.append("STATE " + " ".join(str(int(v)) for v in row))
    lines.append("END")
    print(f"thrml {w['id']} seed {seed} {seconds:.3f}s", file=sys.stderr)
    return lines


def run_workload(w: dict) -> list[str]:
    program, init, observe, decode = _build(w)
    key = jax.random.key(0)
    t0 = time.perf_counter()
    burned = sample_states(key, program, SamplingSchedule(1, 2, 1), init, [], observe)
    jax.block_until_ready(burned)
    jit_s = time.perf_counter() - t0
    lines = []
    for trial in range(w["trials"]):
        lines.extend(_one_trial(w, program, init, observe, decode, w["seed0"] + trial, jit_s))
    return lines


def _nodes_spin(n: int):
    return [SpinNode() for _ in range(n)]


def _blocks(nodes, groups):
    return [Block([nodes[i] for i in g]) for g in groups]


def _ising_program(w: dict):
    nodes = _nodes_spin(w["n"])
    edges = [(nodes[int(e[0])], nodes[int(e[1])]) for e in w["edges"]]
    biases = jnp.asarray(w["biases"], dtype=jnp.float32)
    weights = jnp.asarray([e[2] for e in w["edges"]], dtype=jnp.float32)
    beta = jnp.asarray(w["beta"], dtype=jnp.float32)
    model = IsingEBM(nodes, edges, biases, weights, beta)
    free = _blocks(nodes, w["blocks"])
    program = IsingSamplingProgram(model, free, [])
    init = [jnp.zeros((len(b.nodes),), dtype=bool) for b in free]
    observe = [Block(nodes)]
    return program, init, observe, _decode_bool


def _potts_program(w: dict):
    q = w["q"]
    n = w["n"]
    nodes = [CategoricalNode() for _ in range(n)]
    edges = w["edges"]
    if not edges:
        raise SystemExit(f"{w['id']}: a Potts model with no edges is not the pinned case")
    weights = jnp.zeros((len(edges), q, q), dtype=jnp.float32)
    for e, edge in enumerate(edges):
        for a in range(q):
            weights = weights.at[e, a, a].set(w["beta"] * edge[2])
    factor = CategoricalEBMFactor(
        [Block([nodes[int(e[0])] for e in edges]), Block([nodes[int(e[1])] for e in edges])],
        weights,
    )
    free = _blocks(nodes, w["blocks"])
    spec = BlockGibbsSpec(free, [])
    sampler = CategoricalGibbsConditional(q)
    program = FactorSamplingProgram(spec, [sampler for _ in free], [factor], [])
    init = [jnp.zeros((len(b.nodes),), dtype=jnp.uint8) for b in free]
    return program, init, [Block(nodes)], _decode_int


def _hubo_program(w: dict):
    nodes = _nodes_spin(w["n"])
    beta = w["beta"]
    factors = []
    biases = jnp.asarray([beta * b for b in w["biases"]], dtype=jnp.float32)
    factors.append(SpinEBMFactor([Block(nodes)], biases))
    if w["edges"]:
        left = Block([nodes[int(e[0])] for e in w["edges"]])
        right = Block([nodes[int(e[1])] for e in w["edges"]])
        weights = jnp.asarray([beta * e[2] for e in w["edges"]], dtype=jnp.float32)
        factors.append(SpinEBMFactor([left, right], weights))
    if w["triples"]:
        a = Block([nodes[int(t[0])] for t in w["triples"]])
        b = Block([nodes[int(t[1])] for t in w["triples"]])
        c = Block([nodes[int(t[2])] for t in w["triples"]])
        weights = jnp.asarray([beta * t[3] for t in w["triples"]], dtype=jnp.float32)
        factors.append(SpinEBMFactor([a, b, c], weights))
    free = _blocks(nodes, w["blocks"])
    spec = BlockGibbsSpec(free, [])
    program = FactorSamplingProgram(spec, [SpinGibbsConditional() for _ in free], factors, [])
    init = [jnp.zeros((len(b.nodes),), dtype=bool) for b in free]
    return program, init, [Block(nodes)], _decode_bool


def _decode_bool(samples) -> list[list[int]]:
    arr = jnp.asarray(samples[0])
    # True -> +1, False -> -1. THRML's BernoulliConditional documents that mapping.
    spins = jnp.where(arr, 1, -1)
    return spins.tolist()


def _decode_int(samples) -> list[list[int]]:
    return jnp.asarray(samples[0]).astype(jnp.int32).tolist()


if __name__ == "__main__":
    sys.exit(main())
