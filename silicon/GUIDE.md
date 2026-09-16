# Putting a sampler on silicon

A lab guide for the ferrotherm demo boards. You will place a fabric of stochastic neurons into
real logic cells, wire them through real interconnect, and configure the chip — from a Rust program,
with no vendor toolchain in the path.

Work through it at a terminal with a board plugged in. Each section asks you to predict an outcome
before you run the command. The gap between your prediction and the result is where the learning is.

## Start here

```sh
git clone https://github.com/f4pga/prjxray-db
cargo run --release -p ferrotherm-silicon --example lab -- prjxray-db 64
```

That is the whole path in one command: it loads the fabric map, places 64 neurons, routes their
couplings, checks for shorted wires, and writes a bitstream. It ends by printing the command that
puts the result on the board.

Run it now, before reading further. The rest of this guide explains what each of its six stages
did and why each one can go wrong — which is easier to follow once you have seen the output.

If a database file is missing it says which one and what it was for, rather than failing with a
path error.

---

## What will you have built by the end?

A chip that is sampling. Not simulating a sampler — a physical device whose lookup tables hold the
conditional distributions you chose, whose wires carry one neuron's state to the next, and whose
DONE pin is high because it accepted your configuration.

You will also be able to answer, in joules, what that costs.

## Why can a library configure an FPGA without the vendor's tools?

Because a bitstream is data, and the map is published.

An FPGA is a grid of small configurable pieces: lookup tables that can be any Boolean function of
their inputs, flip-flops, and a switch network joining them. Configuring one means setting several
million bits that say *which* function each lookup table computes and *which* switches are closed.
Those bits are shipped to the chip as a stream of ordinary write commands to a configuration port.

The only hard part is knowing which bit is which. That map was reverse-engineered by
[Project X-Ray](https://github.com/f4pga/prjxray-db) and published as open data. With the map in
hand, generating a bitstream is arithmetic — and arithmetic is something a library can do.

**Get the map:**

```sh
git clone https://github.com/f4pga/prjxray-db
brew install openfpgaloader          # the only other thing you need
```

## What is actually on the chip?

Four things, and you will touch all four.

| Piece | What it does | On an XC7A100T |
|---|---|---|
| **Slice** | Holds four 6-input lookup tables and eight flip-flops | 15,850 |
| **Lookup table** | Computes any function of 6 inputs, from a 64-bit truth table | 63,400 |
| **Switchbox** | Joins wires — an `INT_L` or `INT_R` tile | ~10,350 |
| **Frame** | The unit the configuration port writes: 101 words | ~22,000 |

**Predict first:** a slice holds four lookup tables and there are 15,850 slices. How many lookup
tables is that, and does it match the number in the datasheet?

```rust
use ferrotherm_silicon::tilegrid::TileGrid;

let grid = TileGrid::parse(&std::fs::read_to_string(
    "prjxray-db/artix7/xc7a100t/tilegrid.json")?)?;
println!("{} tiles", grid.tiles.len());
```

## How does a lookup table become a neuron?

A binary stochastic neuron fires with a probability that depends on its input. Written as hardware,
that is a threshold on a count plus a coin:

```text
fire  if  popcount(neighbours) + coin  >=  threshold
```

A 6-input lookup table has exactly 64 possible input patterns, so this rule *is* a 64-bit number —
one bit per pattern, saying whether the neuron fires for it. Nothing is computed at run time; the
answer for every case is stored.

**Predict first:** with five neighbour inputs and one coin input, at threshold 3 — of the 64
patterns, how many should fire?

```rust
use ferrotherm_silicon::lut::bsn_threshold_init;
println!("0x{:016x}", bsn_threshold_init(3));   // count the set bits
```

**What this cell can and cannot express.** Five inputs go to neighbours and one to the coin, so a
neuron has five neighbours and its couplings are unweighted. The only tunable is the integer
threshold, which acts as its field. A spin glass needs weighted couplings, so a spin glass does not
fit this cell — and no amount of spare capacity changes that, because it is a property of the cell,
not of the budget.

**The lever:** spend more than one lookup table per neuron. Weights become expressible as soon as
you are willing to pay in area, which is exactly the trade the [`pdit`](../src/pdit.rs) module
measures for radix-3 units.

## How do two neurons talk?

Through switches. A lookup table's output leaves its slice on a wire, crosses one or more
switchboxes, and arrives at another slice's input. Each switch it passes through is one
configuration bit you must set.

```rust
let path = fabric.route(&source, &destination, 3_000_000, &allow)?;
println!("{} switches", path.len());
```

**Predict first:** two neurons in slices one row apart. How many switches do you expect between
them — one, three, a dozen?

**Why some routes are longer than they look.** The switch network is not uniform. Every fifty rows
of logic, an interconnect column is interrupted by a clock row, and a route crossing that boundary
takes a different path than one inside a region. This is worth knowing before you measure anything:
if a small, *non-random* fraction of your routes behaves differently from the rest, you are looking
at a structural boundary, not at noise.

## Can two neurons share a wire?

No — and the chip will not tell you.

A wire driven by two different outputs has two sources fighting over one conductor. On a CPU that
is a compile error. On an FPGA it is a current. The router finds each path independently and has no
memory between calls, so nothing stops it handing the same switch to two nets.

**Check before you write anything:**

```rust
let clashes = ferrotherm_silicon::route::contentions(&nets);
assert!(clashes.is_empty(), "{clashes:?}");
```

**The general principle:** when a failure mode is invisible to the system that produces it, the
check belongs in the producer. There is no downstream test that recovers this one.

## Where does the configuration actually go?

Into frames, by address. A switch or a truth-table bit resolves to a triple:

```text
frame address  =  the tile's base address + which frame in its column
word in frame  =  the tile's offset + bit / 32
bit in word    =  bit % 32
```

Get that arithmetic wrong and the bitstream is still well-formed. It still passes its checksum. It
still reads back exactly as written — because a reader built from the same table as the writer
agrees with it wherever it is wrong.

**So how do you know the map is right?** Ask something outside your own code. If you have access to
a vendor flow on any machine, it will emit a logic-location file listing the frame address it chose
for every cell it used:

```rust
use ferrotherm_silicon::logic_location::{parse_ll, cross_check};
let report = cross_check(&grid, &parse_ll(&text));
assert!(report.verified());
```

`verified()` requires at least one agreement, not merely an absence of disagreements. A check that
passes when nothing was compared is not a check.

## How do you put it on the board?

```sh
openFPGALoader -c ft2232 --fpga-part xc7a100tfgg484 bsn_fabric.bit
```

```text
Load SRAM: [==================================================] 100.00%
ir: 1 isc_done 1 isc_ena 0 init 1 done 1
```

`done 1` is the DONE pin. The chip took the configuration and finished startup.

**On a Kria KV260** the route is the kernel's FPGA manager instead:

```sh
sudo cp design.bit.bin /lib/firmware/
echo design.bit.bin | sudo tee /sys/class/fpga_manager/fpga0/firmware
sudo dmesg | grep "writing .* to Xilinx ZynqMP FPGA Manager"
```

The last line is not optional. If the file is not in `/lib/firmware`, the write fails, the previous
design keeps running, and `state` still reads `operating`. The kernel log is the only place the
truth appears.

## How do you measure what it costs?

Two things have to be true before a number means anything.

**The design must be reachable.** Synthesis deletes logic that cannot affect any output, and a
sampler's state is read by nothing. An unwrapped fabric optimises down to almost nothing, and the
board then correctly measures near-zero power for a design that is not there. Fold every state bit
into one registered bit, drive something with it, and mark both `DONT_TOUCH`.

**The instrument must respond.** Before trusting a sensor, move it with a load you already
understand. On the KV260, spinning its four processor cores shifts the rail by `+0.681 W` against a
baseline that returns to within `0.008 W`.

Then measure A, B, A — and reload the first design at the end:

| pass | fabric | watts |
|---|---|---|
| 1 | p-bit, 1,024 spins | 3.870742 |
| 2 | p-trit, 64 sites | 4.011461 |
| 3 | p-bit, 1,024 spins | 3.877865 |

The first design reproducing to 0.18% is what makes the middle row a measurement instead of a
drift. Without that third pass you cannot tell the two apart.

**Predict first:** at 100 MHz with two colour classes, a sweep takes two clocks. The p-bit fabric
has 1,024 spins and the p-trit fabric 64 sites. Which fabric performs more updates per second, and
by what factor?

<details>
<summary>The answer, and what it makes the units cost</summary>

The p-bit fabric runs `5.12e10` updates a second, the p-trit `3.2e9` — sixteen times more. Dividing
the power each adds over the common reference:

| | per update | per unit, lookup tables |
|---|---|---|
| p-bit | 7.8 pJ | 43.6 |
| p-trit | 168 pJ | 628 |

A p-trit update costs **21.5 times** a p-bit update, where the area ratio is 14.4. The extra comes
from the three random draws and the multiplexer tree, which switch on every update, while a lookup
table holding a truth table does not.

Both figures share one reference, so the *ratio* is sound. The absolute values are lower bounds:
the reference had a previous design loaded rather than an idle fabric.
</details>

## When something is wrong, what does it look like?

Every entry here produces a plausible result rather than an error, which is why each has a check
rather than a symptom to wait for.

| What you see | What it means | The check |
|---|---|---|
| The IDCODE does not match your constant | The top four bits are a silicon revision | Compare the low 28 bits only |
| A few routes fail, most succeed | A structural boundary, not congestion | Which region do the failures cross? |
| Nothing — but current is high | Two nets on one wire | `contentions()` before writing |
| The loader crashes on a valid stream | A raw image under a `.bit` name | `write_bit` for a container, or name it `.bin` |
| Flash succeeds, behaviour unchanged | The file was not in `/lib/firmware` | `dmesg`, never `state` |
| The board measures ~0 W | The fabric was optimised away | `DONT_TOUCH` and an observable sink |
| A verifier that always passes | Nothing was compared | Require `agreed > 0` |

## Exercises

1. **Place it elsewhere.** Route the same chain down a different column. Predict whether the
   switch count per coupling goes up or down, then count.
2. **Find the boundary.** Chain neurons down a column and increase the length. Predict at what
   length a route first has to cross a clock region, then find it.
3. **Make a contention on purpose.** Route two couplings to the same input wire. Confirm
   `contentions()` names both nets — and do not load the result.
4. **Price a variable.** Using 43.6 lookup tables and 7.8 pJ for a p-bit against 628 and 168 pJ for
   a p-trit, work out the cheaper way to hold one three-state variable. Then read
   [`pdit`](../src/pdit.rs) on why the dearer one can still be the right choice.

## Where to go next

| | |
|---|---|
| The whole path in one command | [`examples/lab.rs`](examples/lab.rs) |
| The same path written out stage by stage | [`examples/bsn_fabric.rs`](examples/bsn_fabric.rs) |
| What a given board can and cannot do | [`ferrotherm::fabric`](../src/fabric.rs) |
| Why a radix-3 unit exists at all | [`ferrotherm::pdit`](../src/pdit.rs) |
| What a p-bit costs, in joules | [`ferrotherm::ledger`](../src/ledger.rs) |
