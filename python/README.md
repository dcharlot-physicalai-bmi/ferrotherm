# ferrotherm

Thermodynamic computing from Python: state a problem, get named values back.

The physics is open and old — Ising (1925), Glauber dynamics (1963), Gibbs sampling (Geman & Geman
1984) — and a "thermodynamic sampling unit" accelerates exactly these loops. This package binds the
[`ferrotherm`](https://crates.io/crates/ferrotherm) Rust library through its C ABI: the solver is
zero-dependency Rust, deterministic by seed, verified against exact physics.

```sh
pip install ferrotherm
```

## State a problem in the vocabulary of the problem

Not a QUBO matrix. Variables have names and domains, constraints say what they mean, and the answer
comes back keyed by the names you used.

```python
import ferrotherm as ft

p = ft.Problem()
a = p.categorical("a", 3)
b = p.categorical("b", 3)
p.not_equal(a, b)
p.maximize(3 * a.is_(1) + 4 * b.is_(2))

ans = p.solve(tries=32)
print(ans.values)      # {'a': 1, 'b': 2}
print(ans.feasible)    # True
```

A constraint can be **soft** — a price rather than a wall — and the answer reports what it cost:

```python
import ferrotherm as ft

p = ft.Problem()
x = p.categorical("x", 3)
y = p.categorical("y", 3)
p.not_equal(x, y, soft=2.0)     # breaking this costs 2.0 x amount squared
ans = p.solve(tries=32)
print(ans.soft_cost, ans.violated)
```

## Or sample the physics directly

```python
import ferrotherm as ft

s = ft.lattice2d(64, 1.0, beta=0.6)   # a magnet below critical temperature
s.sweep(2000)
print(f"|M| = {abs(s.magnetization):.4f}")        # 0.9736
print(f"Onsager = {ft.onsager(0.6):.4f}")         # 0.9736 — the closed form, matched to 4 dp
```

That second line is the check that matters: Onsager's 1944 closed form for the infinite 2D lattice,
and a 64x64 sampler reproducing it to four decimals. The size is not decoration — 16x16 over 500
sweeps gives 0.898 against 0.974, because a small lattice sampled briefly is not the thermodynamic
limit. Verify against exact physics before opinions.

`ft.ring`, `ft.z1_grid`, `ft.frustrated` and `ft.wishart` build the other standard graphs, and
`ft.Model` takes an explicit edge list when you want to write the couplings yourself.

## Every run carries its joules

The ledger is not an appendix. A sampler counts the work it did and prices it at Z1-class device
figures, so the energy question is answerable rather than rhetorical. Note what is being priced:
the modelled device, not the CPU this ran on.

```python
import ferrotherm as ft

s = ft.lattice2d(64, 1.0, beta=0.6)
s.sweep(2000)
print(f"{s.node_updates} node updates -> {s.joules:.2e} J")
```

## The library it binds

This package carries no solver of its own — it loads the shared library and calls it. In order:

1. `FERROTHERM_LIB`, if you set it. An explicit path always wins.
2. The copy shipped in the wheel.
3. `target/release/` of a checkout, for developing against a local build.

`ft.library_path()` tells you which one answered.

Apache-2.0. Source, the Rust crate and eight other bindings:
<https://github.com/dcharlot-physicalai-bmi/ferrotherm>. From the Institute for Physical AI @ BMI.
