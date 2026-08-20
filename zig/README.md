# ferrotherm, from Zig

Thermodynamic computing: state a problem, get named values back.

This is a thin Zig binding over [`ferrotherm`](https://crates.io/crates/ferrotherm)'s C ABI. The
solver is zero-dependency Rust — sparse energy-based models, chromatic block-Gibbs, parallel
tempering, and a joules ledger — verified against exact physics.

## Building

There is no package to fetch. Point Zig at the header and the shared library:

```sh
cargo build --release                    # in the repository root
zig run your.zig -I include -L target/release -lferrotherm -lc
```

`build.zig` here does the same for the module form.

## Using it

State a problem in the vocabulary of the problem, and read the answer back by name:

```zig
const std = @import("std");
const ft = @import("ferrotherm.zig");

pub fn main() !void {
    var p = try ft.Problem.init();
    defer p.deinit();
    const a = try p.categorical("a", 3);
    const b = try p.categorical("b", 3);
    try p.notEqual(a, b);
    try p.prefer(.maximize, 3.0, a.is(1));
    try p.prefer(.maximize, 4.0, b.is(2));
    try p.solve(32);
    std.debug.print("a = {?d}, b = {?d}, feasible = {}\n",
        .{ try p.value(a), try p.value(b), p.feasible() });  // a = 1, b = 2, feasible = true
}
```

`solve` compiles for you, as the Python and Julia bindings do. `compile` is public if you want the
spin count before committing to a run.

Or sample the physics directly and check it against the closed form:

```zig
const s = try ft.Sim.lattice2d(64, 1.0, 0.6, 0);
defer s.deinit();
_ = s.sweep(2000);
std.debug.print("|M| = {d:.4}  Onsager = {d:.4}\n",
    .{ @abs(s.magnetization()), ft.onsager(0.6) });          // 0.9736  0.9736
std.debug.print("{d} node updates -> {e:.2} J\n", .{ s.nodeUpdates(), s.joules() });
```

That comparison is the check that matters: Onsager's 1944 closed form for the infinite 2D lattice,
reproduced to four decimals. The size is not decoration — 16×16 over 500 sweeps gives 0.898 against
0.974, because a small lattice sampled briefly is not the thermodynamic limit.

## What is checked

`scripts/check-parity.sh` proves every C ABI symbol reaches this binding, and
`scripts/check-semantics.sh` proves it compiles one model to **byte-identical** `.ftp` output against
the Rust reference and five other surfaces. A binding that merely exposes the right names is not the
same as one that builds the same model.

Apache-2.0. From the Institute for Physical AI @ BMI.
