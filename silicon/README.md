# ferrotherm-silicon

The IPAI-owned path from a verified ferrotherm sampling fabric to real FPGA silicon, with no
vendor toolchain and no third-party software stack. Pure Rust; the only dependency is the
`ferrotherm` core crate (same repository, same owner).

Selected algorithms are ported, with permission, from Open Interface Engineering's openie-fpga
project. The port is a full re-implementation under IPAI @ BMI's control: no code path in this
crate calls, links, or depends on any external entity's software.

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
