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

# A is the pinned release. B is Extropic's main at this commit, installed from git
# into its own venv. The two are never interchangeable: B's IsingSamplingProgram is a
# logical program, and #77 rewrote the Ising path.
B_COMMIT = "2d96f4b4b67bfa2625d89ed79f0a22c08af3b992"
BASELINE = "A"
from thrml import Block, CategoricalNode, SamplingSchedule, SpinNode, sample_states
from thrml.block_management import block_state_to_global, from_global_state
from thrml.block_sampling import BlockGibbsSpec, sample_with_observation
from thrml.factor import FactorSamplingProgram
from thrml.observers import AbstractObserver
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
    ap.add_argument("--baseline", default="A", choices=["A", "B"])
    args = ap.parse_args()
    global BASELINE
    BASELINE = args.baseline
    try:
        pin = assert_baseline(args.baseline)
    except SystemExit:
        raise
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
        "META thrml={thrml} jax={jaxv} jaxlib={jaxlib} equinox={eqx} device={dev} jax_platform={plat} baseline={base} pin={pin}".format(
            thrml=thrml.__version__,
            jaxv=jax.__version__,
            jaxlib=_jaxlib_version(),
            eqx=_eqx_version(),
            dev=str(device),
            plat=platform,
            base=args.baseline,
            pin=pin,
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
        elif head == "GRID":
            cur["grid"] = dict(part.split("=", 1) for part in rest.split())
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
        "grid": cur.get("grid"),
    }


def bind_device(requested: str):
    """Pick a JAX device. `cuda` requires platform `gpu` and does not fall back to CPU."""
    devices = list(jax.devices())
    platform = choose_platform(requested, [d.platform for d in devices])
    device = next(d for d in devices if d.platform == platform)
    return platform, device


def serve(workloads: list[dict], device, platform: str) -> int:
    """JIT once, then sample one seed per command. QUIT ends the process.

    COST is sweeps with no state and no energy, for the joule window.
    ESS is the energy trace, outside that window.
    RUN is the accuracy trace: states, or energies when the workload is a GRID.
    A workload that fails to build is an ERR for that id. The others still run.
    """
    ready = {}
    failed = {}
    with jax.default_device(device):
        for w in workloads:
            try:
                built = _build(w)
                _warmup_program(w, built)
                ready[w["id"]] = (w, built)
            except MemoryError as err:
                failed[w["id"]] = f"LIMIT oom {err}"
            except Exception as err:
                failed[w["id"]] = f"{type(err).__name__}: {err}"
        print(f"READY platform={platform} baseline={BASELINE}", flush=True)
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            if line == "QUIT":
                return 0
            parts = line.split()
            if len(parts) != 3 or parts[0] not in ("RUN", "COST", "ESS"):
                print(f"ERR unrecognised {line}", flush=True)
                continue
            wid, seed_s = parts[1], parts[2]
            if wid in failed:
                msg = failed[wid]
                if msg.startswith("LIMIT "):
                    print(f"LIMIT {wid} oom", flush=True)
                else:
                    print(f"ERR {wid} {msg}", flush=True)
                continue
            if wid not in ready:
                print(f"ERR unknown workload {wid}", flush=True)
                continue
            w, built = ready[wid]
            seed = int(seed_s)
            mode = {"RUN": "trace", "COST": "cost", "ESS": "ess"}[parts[0]]
            try:
                lines = _dispatch(w, built, seed, mode)
            except MemoryError as err:
                print(f"LIMIT {wid} oom {err}", flush=True)
                continue
            except Exception as err:
                print(f"ERR {wid} {type(err).__name__}: {err}", flush=True)
                continue
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
    try:
        built = _build(w)
        jit_s = _warmup_program(w, built)
    except MemoryError as err:
        return [f"LIMIT {w['id']} oom {err}"]
    except Exception as err:
        raise SystemExit(f"{w['id']}: {type(err).__name__}: {err}") from err
    lines = []
    for trial in range(w["trials"]):
        lines.extend(_dispatch(w, built, w["seed0"] + trial, "trace", jit_s))
    return lines


def _nodes_spin(n: int):
    return [SpinNode() for _ in range(n)]


def _blocks(nodes, groups):
    return [Block([nodes[i] for i in g]) for g in groups]


def _ising_program(w: dict):
    if w.get("grid"):
        return _grid_ising_program(w)
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


def assert_baseline(name: str) -> str:
    """Refuse the wrong THRML. A CPU build is not reported as B, and B is not reported as 0.1.4."""
    from thrml.factor import FactorSamplingProgram
    from thrml.models.ising import IsingSamplingProgram

    executable = issubclass(IsingSamplingProgram, FactorSamplingProgram)
    if name == "A":
        if thrml.__version__ != "0.1.4" or not executable:
            raise SystemExit(
                "baseline A is thrml==0.1.4, whose IsingSamplingProgram is a FactorSamplingProgram. "
                f"Found version {thrml.__version__}, executable={executable}."
            )
        return "0.1.4"
    if name == "B":
        if executable:
            raise SystemExit(
                "baseline B is thrml-main-2d96f4b4. This environment's IsingSamplingProgram is a "
                "FactorSamplingProgram, which is the 0.1.4 shape, not the logical program from #77."
            )
        commit = _vcs_commit()
        if commit != B_COMMIT:
            raise SystemExit(
                f"baseline B requires commit {B_COMMIT}, found {commit or 'no direct_url.json commit'}."
            )
        return commit
    raise SystemExit(f"baseline must be A or B, got {name}")


def _vcs_commit() -> str:
    import importlib.metadata
    import json

    try:
        text = importlib.metadata.distribution("thrml").read_text("direct_url.json")
    except Exception:
        return ""
    if not text:
        return ""
    info = json.loads(text)
    vcs = info.get("vcs_info") or {}
    return str(vcs.get("commit_id") or "")


def _warmup_program(w: dict, built) -> float:
    program, init, observe, _decode = built
    key = jax.random.key(0)
    t0 = time.perf_counter()
    if w.get("grid"):
        _grid_cost(w, program, init, key)
    else:
        burned = sample_states(key, program, SamplingSchedule(1, 2, 1), init, [], observe)
        jax.block_until_ready(burned)
    return time.perf_counter() - t0


def _dispatch(w: dict, built, seed: int, mode: str, jit_s: float = 0.0) -> list[str]:
    program, init, observe, decode = built
    if mode == "cost":
        return _cost_lines(w, program, init, seed)
    if w.get("grid") and mode in ("trace", "ess"):
        return _grid_energy_lines(w, program, init, seed, jit_s)
    return _one_trial(w, program, init, observe, decode, seed, jit_s)


def _cost_lines(w: dict, program, init, seed: int) -> list[str]:
    key = jax.random.key(seed)
    t0 = time.perf_counter()
    sweeps = _grid_cost(w, program, init, key) if w.get("grid") else _state_cost(w, program, init, key)
    seconds = time.perf_counter() - t0
    return [f"DONE id={w['id']} seed={seed} seconds={seconds:.9f} sweeps={sweeps}"]


def _state_cost(w: dict, program, init, key) -> int:
    thin = max(1, w["thin"])
    schedule = SamplingSchedule(w["warmup"] + thin, w["draws"], thin)
    samples = sample_states(key, program, schedule, init, [], [Block([])])
    jax.block_until_ready(samples)
    return w["warmup"] + w["draws"] * thin


def _grid_cost(w: dict, program, init, key) -> int:
    thin = max(1, w["thin"])
    schedule = SamplingSchedule(w["warmup"] + thin, w["draws"], thin)
    observer = NullObserver()
    out = _observe(key, program, schedule, init, observer)
    jax.block_until_ready(out)
    return w["warmup"] + w["draws"] * thin


def _grid_energy_lines(w: dict, program, init, seed: int, jit_s: float) -> list[str]:
    thin = max(1, w["thin"])
    schedule = SamplingSchedule(w["warmup"] + thin, w["draws"], thin)
    side = int(w["grid"]["side"])
    coupling = float(w["grid"]["j"])
    nodes = _grid_nodes_cached(w)
    observer = GridEnergyObserver(blocks_to_sample=[Block(nodes)], side=side, coupling=coupling)
    key = jax.random.key(seed)
    t0 = time.perf_counter()
    _carry, energies = _observe(key, program, schedule, init, observer)
    jax.block_until_ready(energies)
    seconds = time.perf_counter() - t0
    values = [float(v) for v in jnp.asarray(energies).reshape(-1).tolist()]
    if len(values) != w["draws"]:
        raise SystemExit(f"{w['id']} seed {seed}: {len(values)} energies, expected {w['draws']}")
    lines = [
        f"WORK id={w['id']} seed={seed} kind={w['kind']} seconds={seconds:.9f} jit_seconds={jit_s:.9f}"
    ]
    for value in values:
        lines.append(f"ENERGY {value:.16e}")
    lines.append("END")
    print(f"thrml {w['id']} seed {seed} {seconds:.3f}s", file=sys.stderr)
    return lines


_GRID_NODES: dict[str, list] = {}


def _grid_nodes_cached(w: dict) -> list:
    if w["id"] not in _GRID_NODES:
        _GRID_NODES[w["id"]] = _nodes_spin(w["n"])
    return _GRID_NODES[w["id"]]


def _grid_ising_program(w: dict):
    grid = w["grid"]
    side = int(grid["side"])
    coupling = float(grid["j"])
    if side * side != w["n"]:
        raise SystemExit(f"{w['id']}: GRID side {side} does not match n {w['n']}")
    if w["edges"]:
        raise SystemExit(f"{w['id']}: GRID and EDGES are both set. One specification, not two.")
    nodes = _grid_nodes_cached(w)
    edges = []
    weights = []
    for y in range(side):
        for x in range(side):
            i = y * side + x
            if x + 1 < side:
                edges.append((nodes[i], nodes[i + 1]))
                weights.append(coupling)
            if y + 1 < side:
                edges.append((nodes[i], nodes[i + side]))
                weights.append(coupling)
    even = []
    odd = []
    for i, node in enumerate(nodes):
        x = i % side
        y = i // side
        if (x + y) % 2 == 0:
            even.append(node)
        else:
            odd.append(node)
    biases = jnp.zeros((w["n"],), dtype=jnp.float32)
    model = IsingEBM(nodes, edges, biases, jnp.asarray(weights, dtype=jnp.float32), jnp.asarray(w["beta"], dtype=jnp.float32))
    free = [Block(even), Block(odd)]
    program = IsingSamplingProgram(model, free, [])
    init = [jnp.zeros((len(b.nodes),), dtype=bool) for b in free]
    observe = [Block(nodes)]
    return program, init, observe, _decode_bool


def _observe(key, program, schedule, init, observer):
    carry = observer.init()
    if BASELINE == "B":
        return sample_with_observation(
            key, program, schedule, init, [], carry, observer, backend=None
        )
    return sample_with_observation(key, program, schedule, init, [], carry, observer)


class NullObserver(AbstractObserver):
    """Records a scalar zero. The sweep still runs. States are not stacked."""

    def __call__(self, program, state_free, state_clamped, carry, iteration):
        return None, jnp.array(0, dtype=jnp.float32)


class GridEnergyObserver(AbstractObserver):
    """Ferrotherm's open-grid Hamiltonian, E = -J Σ s_i s_j, not THRML's beta-scaled energy."""

    blocks_to_sample: list
    side: int
    coupling: float

    def __call__(self, program, state_free, state_clamped, carry, iteration):
        global_state = block_state_to_global(state_free + state_clamped, program.gibbs_spec)
        sampled = from_global_state(global_state, program.gibbs_spec, self.blocks_to_sample)[0]
        spins = sampled.astype(jnp.int32) * 2 - 1
        grid = spins.reshape((self.side, self.side))
        bonds = jnp.sum(grid[:, :-1] * grid[:, 1:]) + jnp.sum(grid[:-1, :] * grid[1:, :])
        return None, -jnp.asarray(self.coupling, dtype=jnp.float64) * bonds.astype(jnp.float64)


if __name__ == "__main__":
    sys.exit(main())
