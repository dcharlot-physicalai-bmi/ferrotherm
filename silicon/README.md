# ferrotherm-silicon

The IPAI-owned path from a verified ferrotherm sampling fabric to real FPGA silicon, with no
vendor toolchain and no third-party software stack. Pure Rust; the only dependency is the
`ferrotherm` core crate (same repository, same owner).

Selected algorithms are ported, with permission, from Open Interface Engineering's openie-fpga
project. The port is a full re-implementation under IPAI @ BMI's control: no code path in this
crate calls, links, or depends on any external entity's software.

## Using it

The declared capabilities need no board attached, which is most of the value: a caller can ask what
rules their program out before buying hardware.

```rust
use ferrotherm_silicon::device::PtV2;

let f = PtV2::describe();
println!("{} | {:?}", f.name, f.topology);          // alchitry-pt-v2 | Degree(5)
println!("max spins: {:?}", f.max_spins);           // Some(63400)
println!("coupling: {:?}", f.coupling_range);       // integers 0..=1
```

`Degree(5)` is not a simplification: a LUT6 has six inputs and the random bit takes one, so five
remain for neighbours. `63400` is the XC7A100T's LUT6 count. Both are properties of the part, and a
program that needs more of either is refused by `Fabric::check` before anything is flashed.

The LUT arithmetic is public too — a binary stochastic neuron is a threshold on a population count,
and that is a table you can print:

```rust
use ferrotherm_silicon::lut::{bsn_threshold_init, bsn_fire_prob};

println!("0x{:016x}", bsn_threshold_init(3));   // 0xfffefee8fee8e880
println!("{:.4}", bsn_fire_prob(3, 5));         // 1.0000
```


## Verified capability (all measured, not asserted)

| capability | evidence |
|---|---|
| Identify a board | IDCODE 0x13631093 = XC7A100T on a live Alchitry Pt V2 |
| Read configuration registers | STAT/BOOTSTS; IDCODE read by TWO independent paths (JTAG DR and a type-1 config read) agree exactly |
| **Configure real silicon** | 104,140-byte bitstream loaded over JTAG; fabric cleared to 0x00000000 then came back configured |
| Fabric map | 30,932-tile tilegrid parsed in 0.02 s; 15,850 SLICE sites x 4 = 63,400 LUT6 = the datasheet figure |
| Address arithmetic | segbit -> (frame, word, bit), including the multi-word tiles the naive reading misplaces |
| PIP database | 3,629/3,629 non-pseudo INT_L PIPs resolve to bits; the reversed key resolves 7 |
| **Route a signal** | LOGIC_OUTS_L0 (INT_L_X0Y102) -> IMUX3 (INT_R_X1Y102) over an east single-length wire; both PIPs -> physical bits |
| Assemble a bitstream | generated header word-identical to a stream produced for this exact part |

Not yet done: binding a p-bit fabric's LUTs to SLICE site pins (the CLB-to-interconnect node
model), and reading fabric state back (CAPTURE + FDRO) so a running sampler can be observed
without I/O pins.

## Layers (build order = board-readiness order)

1. `lut` — p-bit LUT truth-table generators (the compact 1-LUT popcount-threshold p-bit and the
   LFSR-feedback tables), with exhaustive-enumeration tests, plus the popcount-mode contract for
   `ferrotherm::hdl`'s bit-exact emulator.
2. `ice40` — chip database subset + bitstream emission for the iCE40-HX8K (Alchitry Cu V2): the
   first no-vendor-tools rung.
3. `xc7` — Artix-7 (Au V2, Pt V2, Numato Aller).
4. `zynqusp` — Kria KV260 (LUT/PIP writers, frame addressing, CRC).
5. `vu47p` — Alveo U55C / AWS F2 custom-logic region.
6. `flash` — board programming: FT2232/JTAG, DFU, and the AWS AFI flow.

Every layer holds the ferrotherm verification law: software model == emulator == emitted
configuration, checked by test before anything is flashed.
