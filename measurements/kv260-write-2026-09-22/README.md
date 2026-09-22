# A metered WRITE — Kria KV260, 2026-09-22

Raw sensor data behind `ledger::KV260_WRITABLE_METERED`. The test
`ledger::tests::the_metered_write_is_rederived_from_its_own_sensor_log` parses this file and
recomputes both constants, so the numbers in the crate cannot drift from the evidence.

**This is the first metered write in the crate.** `KV260_AXI_METERED.e_write` is `NaN` and stays
that way: its couplings live in the bitstream, so that design has no write to price. Pricing one
needed a fabric whose couplings are *registers* — `writable::WritableFabric` — and this is the
first time that fabric has run on silicon.

| file | arms | what it prices |
|---|---|---|
| `write_meter_1.dat` | `idle`, `halt`, `write` | a write: `write − idle` over node writes per second; a flip: `idle − halt` over flips per second |

**Design.** `cargo run --release --example board_build -- kv260-axi-w <dir> 16 16 10` — 256 p-bits
(`lattice2d(16)`, β = 0.25) behind `WritableFabric::emit_axi_shell`, couplings as 12-bit signed Q.8
registers, on `M_AXI_HPM0_FPD` at `0xA000_0000`. Vivado 2026.1, `xck26-sfvc784-2LV-c`: 35,152 LUTs,
24,808 registers, **WNS +0.550 ns, 0 failing endpoints of 61,378**, 100 MHz.

**Instrument.** The SOM's INA260 (`ina260_u14`, `hwmon2/power1_input`), 5 V rail, whole board
including regulator loss, sampled at 4 Hz by the same code in every arm.

**Protocol.** 8 passes; arm order shuffled within each pass; 15 s settle then 25 s sampled per arm;
host process pinned to core 3. Statistics are PAIRED: one difference per pass, standard error over
the 8 differences — not over the 768 samples, which share sensor conversion windows.

**Row format.** `S <pass> <arm> <microwatts>` is a sensor sample. `A <pass> arm=… key=value…` is
the host program's own account of that arm. `C …` is the pre-run control.

**Controls with known answers.**
- `C arm=check n=256 pop_bias_max=256 pop_bias_min=0 words_sent=2304 words_accepted=2304`: driving
  every bias to `+max` puts **all 256** spins up and to `−max` puts **all 256** down, so the physics
  follows what was written; and the shell accepted exactly the words the host sent.
- Every running arm reports 24,999,464 sweeps/s against 2.5e7 — 100 MHz at **four clocks a sweep**,
  which is what a writable fabric costs where `FixedFabric` takes two. Measured on silicon, not
  assumed.
- The `halt` arm reports 0.0 sweeps/s and a popcount frozen at one value: held means held.
- `words_accepted == words_sent` in all 8 passes: the write path is lossless at 7.58 M words/s.
- No `# FOREIGN-LOAD` line: nobody else loaded the FPGA during a pass. This board is shared.

**Results.**

| quantity | value |
|---|---|
| `write − idle` | +0.1114 W, se 0.0077, **14.5 σ** |
| write rate | 7,582,370 words/s = 2,527,457 node writes/s (3 words a node) |
| **energy per node write** | **44.07 ± 3.04 nJ** |
| energy per 32-bit config word | 14.69 ± 1.01 nJ |
| `idle − halt` | +0.1225 W, se 0.0052, **23.5 σ** |
| **energy per node update** | **19.145 ± 0.814 pJ** |

**A write costs 2,302 node updates on the same fabric.** That is the number a tempering ladder pays
per node per rung when it rewrites its couplings.

**And the per-beat cost is what dominates, which corrects an easy misreading.** The read metered on
2026-09-19 is quoted as 581 pJ *per spin*, which looks 25× cheaper than this write. It is not: a
read word carries 32 spin values, so per AXI beat that read is `0.0768 W / 4,126,684 = 18.6 nJ`
against this write's `14.7 nJ`. A single-beat AXI4-Lite transaction costs 15–19 nJ on this board in
either direction; 581 pJ is low only because one word carries 32 spins.

**Writability has a price, and it is measured.** 19.145 pJ per node update here against
`KV260_AXI_METERED`'s 9.1316 pJ on the same board and method — 2.1× — for a fabric with 137 LUTs
per p-bit against 44, and four clocks a sweep against two.

## The load that has to be checked, and the wedge it caused

On 2026-09-19 this measurement was attempted and the board wedged, needing a physical power cycle.
The cause is now established. `/sys/class/fpga_manager/fpga0/flags` was **20**; with that set **every
bitstream load errors** — including `ft_axi.bit.bin`, which had loaded successfully hours earlier.
`dmesg` shows `writing <name> to Xilinx ZynqMP FPGA Manager` immediately followed by
`Error while writing image data to FPGA`.

The harness checked that a write had been *attempted*, not that it had *succeeded*, so a failed load
passed the guard, the previous design stayed resident with `state=operating`, and the first
`/dev/mem` access reached an address no slave answered. The core hung in a bus transaction, which is
why the journal for that boot ends with a clean logout and routine housekeeping and then nothing at
all: no oops, no panic, nothing to find afterwards.

`meter_write.py` now refuses when new `Error while writing image data` lines appear, and writes an
fsynced breadcrumb naming the access it is about to make. The guard fired on the first run of this
session and stopped before touching `/dev/mem`. Setting `flags` to 0 makes the load succeed.
