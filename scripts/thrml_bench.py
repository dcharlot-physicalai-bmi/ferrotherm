"""Same-machine, same-model throughput comparison: THRML (JAX) vs ferrotherm.

Builds the identical degree-16 Z1-topology grid at a given size, runs chromatic (2-block) Gibbs
with THRML's own sampling program, and reports flips/s. Run ferrotherm's parity_bench (or the
printed cargo command) at the same size on the same quiet machine for the comparison row.

usage: python scripts/thrml_bench.py [side]   (default 128 -> 128x128 = 16,384 nodes)
"""
import sys, time
import jax
import jax.numpy as jnp
from thrml import Block, SamplingSchedule
from thrml.models.ising import IsingEBM, IsingSamplingProgram
from thrml.block_sampling import sample_states
from thrml.pgm import SpinNode

def build(side):
    n = side * side
    nodes = [SpinNode() for _ in range(n)]
    rules = [(1, 0), (2, 1), (2, 3), (4, 1)]
    edges = []
    for y in range(side):
        for x in range(side):
            i = y * side + x
            for (a, b) in rules:
                for (dx, dy) in [(a, b), (-b, a), (-a, -b), (b, -a)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < side and 0 <= ny < side:
                        j = ny * side + nx
                        if j > i:
                            edges.append((nodes[i], nodes[j]))
    even = Block([nodes[y * side + x] for y in range(side) for x in range(side) if (x + y) % 2 == 0])
    odd = Block([nodes[y * side + x] for y in range(side) for x in range(side) if (x + y) % 2 == 1])
    biases = jnp.zeros((n,))
    weights = jnp.full((len(edges),), 0.08)
    ebm = IsingEBM(nodes, edges, biases, weights, jnp.asarray(0.9))
    prog = IsingSamplingProgram(ebm, [even, odd], [])
    return prog, even, odd, n

def main():
    side = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    t0 = time.time()
    prog, even, odd, n = build(side)
    print(f"built {side}x{side} = {n} nodes in {time.time()-t0:.1f} s (device: {jax.devices()[0].platform})")
    key = jax.random.PRNGKey(42)
    init = [jnp.zeros((len(even.nodes),), dtype=bool), jnp.zeros((len(odd.nodes),), dtype=bool)]
    # warmup: includes JIT compile; excluded from timing
    sched_w = SamplingSchedule(n_warmup=3, n_samples=1, steps_per_sample=1)
    t0 = time.time()
    out = sample_states(key, prog, sched_w, init, [], [even])
    jax.block_until_ready(out)
    print(f"compile + warmup: {time.time()-t0:.1f} s")
    # timed: n_sweeps full sweeps (both blocks per step)
    n_sweeps = 200
    sched = SamplingSchedule(n_warmup=n_sweeps, n_samples=1, steps_per_sample=1)
    t0 = time.time()
    out = sample_states(key, prog, sched, init, [], [even])
    jax.block_until_ready(out)
    dt = time.time() - t0
    fps = n_sweeps * n / dt
    print(f"THRML: {n_sweeps} sweeps of {n} nodes in {dt:.2f} s -> {fps:.3e} flips/s ({1e9/fps:.1f} ns/flip)")
    print(f"compare: cargo run --release --example parity_bench_size -- {side}")

if __name__ == "__main__":
    main()
