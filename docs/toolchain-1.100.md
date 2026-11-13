# Toolchain 1.100

Record of the move onto Rust 1.100.0. `rules_version` stays `v3`.
`METHODOLOGY.md` at `dbf9653a` is unchanged.

## Compiler

Official stable channel manifest `https://static.rust-lang.org/dist/channel-rust-stable.toml`,
fetched 2026-11-13, date `2026-11-12`:

```
rustc 1.100.0 (0ef1bc386 2026-11-09)
binary: rustc
commit-hash: 0ef1bc386059596dada8fb65b50425aab2ea637a
commit-date: 2026-11-09
host: x86_64-unknown-linux-gnu
release: 1.100.0
LLVM version: 23.1.5
```

No newer stable was on the channel. `rust-toolchain.toml` pins `1.100.0`.

`777f0417` does not pin a compiler. Its `rust-version` is `1.96`, and that is
the compiler the parent binary was built with:

```
rustc 1.96.0 (ac68faa20 2026-05-25)
commit-hash: ac68faa20c58cbccd01ee7208bf3b6e93a7d7f96
commit-date: 2026-05-25
LLVM version: 22.1.2
```

`rust-version = "1.96"` stays. It is the policy floor `scripts/check-msrv.sh`
compiles, not the compiler CI uses. `cargo update --ignore-rust-version`
moved zero packages, so the floor is not what is holding a crate back.

## cargo audit

`cargo audit` 0.23.0, advisory db 1509 entries, 178 crate dependencies in
`Cargo.lock`. Exit 0. No advisory was printed. There is nothing to ignore
and nothing upstream-blocked on a RUSTSEC id.

## cargo deny

`cargo deny` 0.20.2, `cargo deny --workspace check`, exit 0.
Advisories, bans, licenses, and sources all ok. The only duplicate-version
skips are `syn` 2/3 and `hashbrown` 0.16/0.17, both required by the latest
`wgpu` 30.0.2 graph. They are named in `deny.toml` with the parent crate.

## Direct dependencies

| Crate | Was | Now | Note |
| --- | --- | --- | --- |
| wgpu | `30` (latest 30.0.2) | 30.0.2 | current major; no 31 |
| pollster | 0.4 | 1.0.1 | API unchanged at the call site (`block_on`) |
| ureq | 2 | 3.5.2 | `Agent` + `header` + `send` + `body_mut().read_to_string` |
| nusb | 0.1 | 0.2.9 | `MaybeFuture::wait`, endpoint `transfer_blocking` |

There was no `Cargo.lock` before this branch. The new lock's sha256 is
`0be76a57fb3feec05afdb6df2e604295c41c536e4e78caa23b0a1609b93b3d32`.

## Transitive crates not at their own newest major

These are not choices in this repo's manifests. The latest release of the
direct crate still depends on the older major. Overriding them would be a
fork of wgpu or ureq, not an upgrade.

| Crate locked | Newest on crates.io | Held by |
| --- | --- | --- |
| syn 2.0.119 (also syn 3.0.10) | 3.0.10 | wgpu 30 → naga → half → zerocopy-derive |
| hashbrown 0.16.1 (also 0.17.1) | 0.17.1 | wgpu 30 → gpu-allocator 0.28 |
| getrandom 0.2.17 | 0.4.3 | ureq 3.5.2 → rustls → ring 0.17 |
| glow 0.17.0 | 0.18.0 | wgpu-hal 30.0.2 |
| libloading 0.8.9 | 0.9.1 | wgpu-hal → ash → gpu-allocator |
| allocator-api2 0.2.21 | 0.4.0 | hashbrown 0.16 → gpu-allocator |
| rustc-hash 1.1.0 | 2.1.3 | naga 30.0.2 |
| parking_lot 0.12.5 / parking_lot_core 0.9.12 / lock_api 0.4.14 | 0.13 / 0.10 / 0.5 | wgpu 30 graph |
| windows-sys 0.52.0 (also 0.61.2) and the 0.52 windows-* targets | 0.61 / 0.53 | older half of the wgpu/nusb windows graph |
| bit-set 0.10.0 / bit-vec 0.9.1 | 0.11.1 / 0.10.2 | parent still on the 0.10 line |
| xml-rs 0.8.29 | 1.0.0 | not selected by a direct manifest |
| wasi 0.11.1 | 0.14.7 | parent still on preview1 |
| jni-sys 0.3.1 (also 0.4.1) | 0.4.1 | one parent still on 0.3 |
| mach2 0.5.0 / redox_syscall 0.5.18 | 0.7.0 / 0.9.5 | platform support crates inside wgpu |

## Accuracy receipts versus 777f0417

Same command, both release binaries, THRML skipped:

```
ferrotherm-thrml-bench --out DIR --skip-thrml --trials 2 --warmup 64 --draws 256
```

Old binary: rustc 1.96.0, `.comment` says so. New binary: rustc 1.100.0.

`spec.txt` byte-identical. sha256
`5b112f0c9de7ba4fc8ff20d000b1e73edf8aa097437d9e3c40d6848435703b6f`.
Both receipts stamp that as `spec_sha256`.

State and energy traces, all eight files, byte-identical.

`report.json` is not byte-identical.

| | old | new |
| --- | --- | --- |
| file sha256 | `d3a9484c6800afd6c0d2c7c8319ec216bd1bb79aaa5261ff1b2c4c8ef6cb9b8c` | `e0f7157baf9b0946b32dc4a67f30e294997edd2fef4f30ec548ea2cb83c7b2eb` |
| `receipt_sha256` | `43aef3cddde655d28d1ef0639bdc0f4deb056b9a1f45e51944ad41106575a8be` | `1cf6158f73bada27a7798865226900c0db684ad7ab015a7f7329eca3900bf994` |

Walking every JSON value, the fields that differ are:

- `receipt_sha256`, because the file bytes differ.
- `provenance.rustc`: one line from `rustc --version` on the parent, `rustc -Vv` on this branch.
- `provenance.cargo_lock_sha256`: added on this branch. The parent has no lock file.
- `host_lock.acquired_utc`, `released_utc`, `holder_pid`: wall clock and pid.
- `seconds`, `wall_ns_per_sweep`, `ess_per_second`, and the means of those rates.

`ess_per_second` is ESS divided by wall seconds. `wall_ns_per_sweep` is elapsed
time. METHODOLOGY v3 already says those timers are not joules and are not the
parity line. No tolerance was widened.

Unchanged, value for value: `tv`, `kl_empirical_exact`, `tau_int_energy`,
`ess_energy`, `ess_per_sweep`, `mean_energy`, `exact_mean_energy`,
`energy_abs_err`, `magnetization_mae`, `missed_states`, `gibbs_updates`,
`reads`, `rules_version` (`v3`).

CPU lever dump (scalar, parallel, simd, tiled, sparse-rng, incremental,
combined, and the packed checkerboard; side 8, seed `260801615+9000*8`,
warmup 400, 4096 draws), f64 bit patterns and final spins:
byte-identical, sha256
`5e17ea2457534ff3c215f29acc779571c1bf0fbe817126930c98ce9397021d9e`.

Nothing in those comparisons was re-pinned.

## What this machine cannot reproduce

Energy receipts that need RAPL, NVML, tegrastats, INA238, or macmon, and any
campaign on jetson-hub or Thor. This VM has no Vulkan ICD listing under
`/usr/share/vulkan/icd.d` and no `iverilog`. The GPU crate's tests still ran
and passed here (`conform_can_finally_score_the_gpu_path` included); that is
not a joule measurement. RTL bit-exact tests that need iverilog were not
re-simulated on this machine.

## THRML

PyPI `thrml` latest is 0.1.4. GitHub releases stop at `v0.1.4` (2026-08-04).
There is no newer release, so there is no baseline C. Baseline A
(`thrml==0.1.4`, JAX `0.12.0.1`) and baseline B (`2d96f4b4`, the same JAX pins)
are unchanged. JAX's current release is `0.12.0.1`, the same pin.
