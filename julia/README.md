# Ferrotherm.jl

Julia bindings to [ferrotherm](https://github.com/dcharlot-physicalai-bmi/ferrotherm), through the
same C ABI every other binding uses.

## Installing

Not from the General registry — see below for exactly why, and what it would take.

```julia
pkg> add https://github.com/dcharlot-physicalai-bmi/ferrotherm.git:julia/Ferrotherm
```

The package needs the shared library. It looks in three places, in order:

1. `ENV["FERROTHERM_LIB"]`, set before `using Ferrotherm` — an explicit path always wins.
2. The artifact from `ferrotherm_jll`, if that package is installed. Discovered at runtime with
   `Base.identify_package`, deliberately **not** declared in `[deps]`, so that a checkout built from
   source does not drag a binary artifact in behind it.
3. The platform library name on the loader's own search path.

Building from source is two commands, and is what the repository's own CI does:

```sh
cargo build --release
export FERROTHERM_LIB=$PWD/target/release/libferrotherm.dylib   # .so on Linux, .dll on Windows
```

## Using it

State a problem in the vocabulary of the problem — named variables, named constraints, and an answer
keyed by the names you used.

```julia
using Ferrotherm

p = Problem()
a = categorical!(p, "a", 3)
b = categorical!(p, "b", 3)
not_equal!(p, a, b)
maximize!(p, [(3.0, is(a, 1)), (4.0, is(b, 2))])

ans = solve!(p; tries = 32)
println(ans.values)        # Dict("b" => 2, "a" => 1)
println(feasible(ans))     # true
```

Or sample the physics directly, and check it against the closed form:

```julia
using Ferrotherm

s = lattice2d(64; J = 1.0, beta = 0.6)   # note the keyword arguments
sweep!(s, 2000)
println(round(abs(magnetization(s)); digits = 4))   # 0.9736
println(round(onsager(0.6); digits = 4))            # 0.9736 — Onsager's closed form
println(node_updates(s), " node updates -> ", joules_z1(s), " J")
```

That second comparison is the check that matters: Onsager's 1944 result for the infinite 2D lattice,
reproduced to four decimals by a 64×64 sampler. The size is not decoration — 16×16 over 500 sweeps
gives 0.898 against 0.974, because a small lattice sampled briefly is not the thermodynamic limit.

The ledger travels with the run: `node_updates` and `joules_z1` price the modelled device, not the
CPU this ran on.

## Why this is not `Pkg.add("Ferrotherm")`

Checked against the registry's own rules on 2026-08-16 rather than assumed, because the answer
changed recently. Two things block it, and they are independent.

**1. The binary. `ferrotherm_jll` can no longer be registered at all.** General's
[`manual_jll.md`](https://github.com/JuliaRegistries/General/blob/master/manual_jll.md) records that
in **February 2026** the maintainers decided to stop accepting manual JLLs. The prohibition names
new packages ending in `_jll` and packages with self-hosted binaries — which is precisely what
`ferrotherm_jll` is. Existing manual JLLs are grandfathered; ours is not one, having never been
registered. Exceptions are case-by-case and are meant for technical or legal barriers to using
Yggdrasil, neither of which applies here.

So the only route for the binary is **[Yggdrasil](https://github.com/JuliaPackaging/Yggdrasil)**, the
community build tree: a BinaryBuilder recipe that cross-compiles this crate's `cdylib` for the
standard platform set, submitted as a pull request. That is real work with an external review, not
a flag to flip. It is also the *right* answer — it buys provenance, reproducible cross-compilation
and the platform matrix, none of which a hand-rolled artifact has.

**2. The package. AutoMerge requires the repository URL to end in `/PackageName.jl.git`.** This is a
monorepo at `/ferrotherm`, and `Ferrotherm.jl` lives in a subdirectory of it, so AutoMerge would
refuse on the URL rule alone. Registering would need a dedicated `Ferrotherm.jl` repository, or a
manual merge with maintainer sign-off. The *name* is fine: `Ferrotherm` starts upper-case, is ASCII
alphanumeric, contains a lower-case letter, exceeds five characters, and does not contain `julia`,
start with `Ju` or end with `jl`.

Registering the package without the binary would be worse than not registering it: `Pkg.add` would
succeed and `using Ferrotherm` would then fail for anyone who had not separately built or fetched a
library. An install that resolves and does not run is not an install.

## What it does

Everything the C ABI exposes, with Julia names — `Problem`, `categorical!`, `integer!`,
`not_equal!`, `at_most!`, `fix!`, `maximize!`, `solve!`, `ftp`. `scripts/check-parity.sh` proves
every C ABI symbol reaches this binding, `scripts/check-exports.sh` proves every exported name
actually resolves to something (both were once declared with no function body in between, and the
module loaded fine), and `scripts/check-semantics.sh` proves this binding compiles a model to
**byte-identical** `.ftp` output against the Rust reference and five other surfaces.

Optional extensions load when their packages are present: `QUBODrivers`/`QUBOTools` for the Julia
optimisation ecosystem, and `Graphs` for graph construction.
