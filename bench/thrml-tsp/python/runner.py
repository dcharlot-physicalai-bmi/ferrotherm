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
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    workloads = parse_spec(open(args.spec, encoding="utf-8").read())
    lines = [
        "META thrml={thrml} jax={jaxv} jaxlib={jaxlib} equinox={eqx} device={dev}".format(
            thrml=thrml.__version__,
            jaxv=jax.__version__,
            jaxlib=_jaxlib_version(),
            eqx=_eqx_version(),
            dev=str(jax.devices()[0]),
        )
    ]
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


def run_workload(w: dict) -> list[str]:
    kind = w["kind"]
    if kind == "ising":
        build = _ising_program
    elif kind == "potts":
        build = _potts_program
    elif kind == "hubo":
        build = _hubo_program
    else:
        raise SystemExit(f"unknown kind {kind}")
    program, init, observe, decode = build(w)
    # Compile on a throwaway schedule. Not included in the sample timing.
    key = jax.random.key(0)
    t0 = time.perf_counter()
    burned = sample_states(key, program, SamplingSchedule(1, 2, 1), init, [], observe)
    jax.block_until_ready(burned)
    jit_s = time.perf_counter() - t0
    lines = []
    thin = max(1, w["thin"])
    schedule = SamplingSchedule(w["warmup"] + thin, w["draws"], thin)
    for trial in range(w["trials"]):
        seed = w["seed0"] + trial
        key = jax.random.key(seed)
        t0 = time.perf_counter()
        samples = sample_states(key, program, schedule, init, [], observe)
        jax.block_until_ready(samples)
        seconds = time.perf_counter() - t0
        states = decode(samples)
        if len(states) != w["draws"]:
            raise SystemExit(f"{w['id']} seed {seed}: expected {w['draws']} states, got {len(states)}")
        width = len(states[0])
        if width != w["n"]:
            raise SystemExit(f"{w['id']} seed {seed}: expected width {w['n']}, got {width}")
        lines.append(
            f"WORK id={w['id']} seed={seed} kind={kind} seconds={seconds:.9f} jit_seconds={jit_s:.9f}"
        )
        for row in states:
            lines.append("STATE " + " ".join(str(int(v)) for v in row))
        lines.append("END")
        print(f"thrml {w['id']} seed {seed} {seconds:.3f}s", file=sys.stderr)
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
