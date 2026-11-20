# Toolchain for campaigns under measurement rules v3.1

Campaigns from this note on build with the compiler pinned in
`rust-toolchain.toml`. The harness rules stay v3.1. This note is the
toolchain and the provenance a receipt records. It does not edit
`docs/METHODOLOGY-v3.1.md` or `bench/thrml-tsp/METHODOLOGY.md`.

Brackets, the 3σ test, ABBA order, the load1 settle, the trial-window
thresholds, the stop rule, and the 0.05 parity line stay the v3.1 text.
THRML A stays `thrml==0.1.4`. THRML B stays `2d96f4b4`. JAX stays
`0.12.0.1`.

## Compiler

Newest stable `rustup` offered on 2026-11-20. The channel manifest
`https://static.rust-lang.org/dist/channel-rust-stable.toml` is dated
`2026-11-12`. No newer stable was on that channel.

```
rustc 1.100.0 (0ef1bc386 2026-11-09)
binary: rustc
commit-hash: 0ef1bc386059596dada8fb65b50425aab2ea637a
commit-date: 2026-11-09
host: x86_64-unknown-linux-gnu
release: 1.100.0
LLVM version: 23.1.5
```

`rust-version = "1.96"` stays. It is the policy floor
`scripts/check-msrv.sh` compiles, not the compiler a campaign uses.

The pin is the rewritten toolchain commit
`8059b4b1605d3c5c0698025c34e8ba90a9ed389d`
(`rewrite/chore/toolchain-1.100`) brought onto
`af24753655c067ac63a7ce50b5bd61af9ced8a28`. Direct crates are pollster
1.0.1, ureq 3.5.2, nusb 0.2.9, and wgpu 30.0.2. `docs/toolchain-1.100.md`
is the lock, audit, and deny record of that pin.

## Equivalence

On jetson-hub and on thor, every deterministic output was byte-identical
between rustc 1.99.1 and rustc 1.100.0: the spec, the energy traces, and
the chain values. Only timing fields and provenance fields differ.

That evidence covers this pin. A toolchain past 1.100.0 is outside it.
Equivalence has to be shown again on that version before a campaign
builds with it.

## Provenance

A receipt records `rustc -Vv` and the sha256 of `Cargo.lock`. Those
fields, and the wall-clock timers, are the fields a compiler change is
allowed to move. Spec bytes, energy-trace bytes, and chain values are
not.

## Checked against af247536 built with rustc 1.96.0

Release binaries on this machine. The parent is
`af24753655c067ac63a7ce50b5bd61af9ced8a28`, rustc 1.96.0
(`ac68faa20`, 2026-05-25). This pin is rustc 1.100.0.

```
ferrotherm-thrml-bench --out DIR --skip-thrml --trials 2 --warmup 64 --draws 256
```

`spec.txt` is byte-identical. sha256
`5b112f0c9de7ba4fc8ff20d000b1e73edf8aa097437d9e3c40d6848435703b6f`.
The eight state and energy traces are byte-identical.

The CPU lever dump is byte-identical. sha256
`68333a71295f83187998035c1764e072d8d5711f9c7204cc95aa1f384bcee00b`.
It is scalar, parallel, simd, tiled, sparse-rng, incremental, combined,
and the packed checkerboard. Side 8, `J = 1`, `beta = 0.3`, seed
`260801615+9000*8`, warmup 400, 4096 draws, 4 threads. The file is the
f64 bit patterns and the final spins.

`report.json` is not byte-identical. What moves is the host lock
(timestamps and pid), the load samples, `receipt_sha256`,
`provenance.git_rev`, `provenance.rustc` (`rustc --version` on the 1.96
binary, `rustc -Vv` on this pin), `provenance.cargo_lock_sha256` (the
lock's sha256 is
`0be76a57fb3feec05afdb6df2e604295c41c536e4e78caa23b0a1609b93b3d32`),
and the timers `seconds`, `wall_ns_per_sweep`, and `ess_per_second`.
`rules_version` stays `v3.1`. The THRML pin string stays
`thrml==0.1.4` and `2d96f4b4b67bfa2625d89ed79f0a22c08af3b992`.
