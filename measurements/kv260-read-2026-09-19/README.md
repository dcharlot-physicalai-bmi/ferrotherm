# A metered read, and a metered flip, on one bitstream — Kria KV260, 2026-09-19

Raw sensor data behind `ledger::KV260_AXI_METERED`. The test
`ledger::tests::the_metered_read_is_rederived_from_its_own_sensor_log` parses these two files and
recomputes both constants, so the numbers in the crate cannot drift from the evidence.

| file | arms | what it prices |
|---|---|---|
| `read_meter_1.dat` | `idle`, `cpu`, `axi` | a read: `axi − idle` over spin values read per second |
| `read_meter_2.dat` | `idle`, `halt` | a flip: `idle − halt` over flips per second |

**Design.** `cargo run --release --example board_build -- kv260-axi <dir> 16 32 10` — 1,024 p-bits
(`lattice2d(32)`, β = 0.6) behind `FixedFabric::emit_axi_shell`, on `M_AXI_HPM0_FPD` at
`0xA000_0000`, 128-bit port to match the running PS (`afi_fs = 0xA00`). Vivado 2026.1,
`xck26-sfvc784-2LV-c`: 45,196 LUTs, 34,821 registers, WNS +4.029 ns at 100 MHz, 0 failing endpoints.

**Instrument.** The SOM's INA260 (`ina260_u14`, `hwmon2/power1_input`), 5 V rail, whole board
including regulator loss, sampled at 4 Hz by the same code in every arm.

**Protocol.** 8 passes; arm order shuffled within each pass; 15 s settle then 25 s sampled per arm;
host process pinned to core 3. Statistics are PAIRED: one difference per pass, standard error over
the 8 differences — not over the 768 samples, which share sensor conversion windows.

**Row format.** `S <pass> <arm> <microwatts>` is a sensor sample. `A <pass> arm=… key=value…` is
the host program's own account of that arm.

**Controls with known answers, all in the `A` rows.**
- Every running arm reports 49,998,9xx sweeps per second against 5.0e7 (100 MHz, two clocks a
  sweep; the PS clock is 20 ppm slow). A port width that disagreed with the PS would fail this.
- The `halt` arm reports 0.0 sweeps per second, a popcount frozen at one value, and zero changes
  of state word 0: held means held.
- No `# FOREIGN-LOAD` line: nobody else loaded the FPGA during a pass. This board is shared.

**Results.**

| quantity | value |
|---|---|
| `axi − idle` | +0.0768 W, se 0.0074, 10.3 σ |
| read rate | 4,126,684 words/s = 1.3205e8 spin values/s, one A53 core, single-beat AXI4-Lite |
| **energy per spin read** | **581 ± 56 pJ** |
| `idle − halt` | +0.4675 W, se 0.0067, 69.4 σ |
| flip rate | 5.1199e10 /s |
| **energy per flip** | **9.13 ± 0.13 pJ** |
| read / flip | 63.7 |
| one full-state readout | 7.75 µs = 388 sweeps |

**A control that FAILED, kept on the record.** The `cpu` arm spins the same loop over the core's
own memory, and the plan was to report `axi − cpu` as the bus-and-fabric share of a read. It came
out **−0.0714 W** (−11.4 σ): a core stalled on AXI draws less than a core spinning in cache. The
prediction that it would be a small positive number was wrong, and the arm bounds nothing. The
read is priced as `axi − idle`, which includes the core that issues it — a read needs a master.

**What these figures are not.** Single-beat AXI4-Lite through a CPU is the most expensive way to
move a bit off this fabric; a burst or DMA path was not built and would be cheaper per bit. The
flip figure is datapath dynamic power only: the clock tree runs in both arms and cancels, which is
why it sits below the 10.85 pJ of 2026-09-06, whose baseline was an idle PL.
