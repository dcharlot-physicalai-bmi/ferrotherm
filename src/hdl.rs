//! The hardware backend: lower a sampling graph to a fixed-point p-bit fabric and emit
//! synthesizable Verilog for it — the same design for every deployment target in [`crate::targets`],
//! from an Alchitry board to an Alveo card to an AWS F2 instance.
//!
//! The contract that makes this trustworthy: [`FixedFabric`] is a CYCLE-EXACT Rust emulator of
//! the emitted hardware — same quantization (Q.8 weights), same sigmoid ROM, same per-node
//! xorshift32 RNG, same two-phase chromatic schedule. The Verilog testbench replays the emulator's
//! per-sweep state trace and must match BIT-EXACTLY in simulation (icarus-verilog gate in the
//! tests). Software model == emulator == RTL, or the build fails.
//!
//! v1 scope: bipartite pairwise spin graphs (the lattice and Z1-topology classes), free-running
//! (no clamp ports yet), fully parallel per color — one full sweep per two clock cycles.

use crate::graph::Graph;

/// Fractional bits in the fixed-point format: Q.8, so one unit is `1/256`.
pub const FRAC: u32 = 8; // Q.8 fixed point
const FMAX: i32 = 2047; // clamp field to [-8.0, +8.0) in Q.8
const FMIN: i32 = -2048;
/// Address width of the sigmoid ROM, so it holds `2^10 = 1024` entries.
pub const LUT_BITS: u32 = 10; // 1024-entry sigmoid ROM

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E3779B97F4A7C15);
    let mut x = z;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

#[inline]
fn xorshift32(mut x: u32) -> u32 {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    x
}

/// A bit-accurate emulator of the generated RTL: same fixed point, same ROM, same RNG.
///
/// It exists so the hardware and the software can be compared state by state rather than in
/// distribution -- a sampler that agrees on averages can still disagree on every step.
pub struct FixedFabric {
    /// Nodes in the fabric.
    pub n: usize,
    /// per node: (neighbor index, quantized weight)
    pub adj: Vec<Vec<(u32, i32)>>,
    /// Per-node bias, quantised to Q.8.
    pub bias_q: Vec<i32>,
    /// the two chromatic classes (v1 requires a bipartite graph)
    pub classes: [Vec<u32>; 2],
    /// The sigmoid ROM the RTL indexes, `2^LUT_BITS` entries.
    pub lut: Vec<u16>,
    /// Per-node RNG seeds, so the emulator and the RTL start from the same streams.
    pub seeds: Vec<u32>,
    /// Initial state the RTL loads on reset (`true` = +1).
    pub init_s: Vec<bool>,
    /// current emulator state (true = +1)
    pub s: Vec<bool>,
    /// Per-node xorshift32 state, advanced once per node per sweep.
    pub rng: Vec<u32>,
}

impl FixedFabric {
    /// Quantize a graph at inverse temperature `beta` into the fabric. Panics unless the
    /// coloring is exactly two classes (the lattice / device-topology case v1 targets).
    ///
    /// # Panics
    ///
    /// If the graph is not bipartite. The v1 fabric updates two colour classes and has nowhere to put
    /// a third.
    #[must_use]
    pub fn new(g: &Graph, beta: f64, seed: u64) -> FixedFabric {
        assert_eq!(g.classes.len(), 2, "v1 fabric requires a bipartite (2-colorable) graph");
        let scale = (1u32 << FRAC) as f64;
        let mut adj = vec![Vec::new(); g.n];
        for i in 0..g.n {
            for k in g.offset[i]..g.offset[i + 1] {
                adj[i].push((g.nbr[k], (g.w[k] * scale).round() as i32));
            }
        }
        let bias_q: Vec<i32> = g.h.iter().map(|&h| (h * scale).round() as i32).collect();
        let lut: Vec<u16> = (0..(1usize << LUT_BITS))
            .map(|a| {
                let arg = ((a as f64 + 0.5) * 4.0 - 2048.0) / scale;
                // Emitted hardware cannot call the Rust kernel, so it must agree with it
                // instead. This is the one place the update is legitimately duplicated, and
                // `lut_agrees_with_the_kernel` below is what keeps the duplicate honest.
                let p = crate::kernel::p_up(arg, beta);
                (p * 65535.0).round().min(65535.0) as u16
            })
            .collect();
        let seeds: Vec<u32> = (0..g.n)
            .map(|i| {
                let s = splitmix(seed ^ (i as u64).wrapping_mul(0xD6E8FEB86659FD93)) as u32;
                if s == 0 { 1 } else { s }
            })
            .collect();
        let init_s: Vec<bool> = (0..g.n).map(|i| splitmix(seed ^ 0xA5A5 ^ i as u64) & 1 == 1).collect();
        FixedFabric {
            n: g.n,
            adj,
            bias_q,
            classes: [g.classes[0].clone(), g.classes[1].clone()],
            lut,
            seeds: seeds.clone(),
            init_s: init_s.clone(),
            s: init_s,
            rng: seeds,
        }
    }

    #[inline]
    fn update_node(&mut self, i: usize) {
        let mut field = self.bias_q[i];
        for &(j, w) in &self.adj[i] {
            field += if self.s[j as usize] { w } else { -w };
        }
        let fc = field.clamp(FMIN, FMAX);
        let addr = ((fc + 2048) >> 2) as usize;
        let p16 = self.lut[addr];
        let nx = xorshift32(self.rng[i]);
        self.rng[i] = nx;
        let rand16 = (nx >> 16) as u16;
        self.s[i] = rand16 < p16;
    }

    /// One full sweep: phase 0 updates class 0 (reading registered class-1 states), then phase 1.
    pub fn sweep(&mut self) {
        for phase in 0..2 {
            let class = self.classes[phase].clone();
            for &iu in &class {
                self.update_node(iu as usize);
            }
        }
    }

    #[must_use]
    /// Mean spin of the current state, in `[-1, 1]`.
    pub fn magnetization(&self) -> f64 {
        let up = self.s.iter().filter(|&&b| b).count() as f64;
        (2.0 * up - self.n as f64) / self.n as f64
    }

    /// Reset the emulator to the exact power-on state of the emitted hardware.
    pub fn reset(&mut self) {
        self.s.copy_from_slice(&self.init_s);
        self.rng.copy_from_slice(&self.seeds);
    }

    /// Emit the synthesizable fabric module.
    #[must_use]
    pub fn emit_verilog(&self, module: &str) -> String {
        let n = self.n;
        let mut v = String::new();
        v.push_str(&format!(
            "// generated by ferrotherm::hdl — fixed-point chromatic-Gibbs p-bit fabric\n\
             // {n} p-bits, Q.{FRAC} weights, {}-entry sigmoid ROM, xorshift32 per node\n\
             module {module} (\n    input wire clk,\n    input wire rst,\n    input wire en,\n    output reg [{top}:0] state,\n    output reg phase\n);\n",
            1usize << LUT_BITS,
            top = n - 1
        ));
        v.push_str("  function [15:0] fsig; input [9:0] a; begin\n    case (a)\n");
        for (a, p) in self.lut.iter().enumerate() {
            v.push_str(&format!("      10'd{a}: fsig = 16'd{p};\n"));
        }
        v.push_str("      default: fsig = 16'd0;\n    endcase\n  end endfunction\n\n");
        v.push_str(
            "  function [31:0] xs32; input [31:0] x; reg [31:0] a, b; begin\n    a = x ^ (x << 13); b = a ^ (a >> 17); xs32 = b ^ (b << 5);\n  end endfunction\n\n",
        );
        v.push_str(&format!("  reg [31:0] rng [0:{}];\n", n - 1));
        for i in 0..n {
            let mut terms = vec![format!("32'sd{}", self.bias_q[i])];
            for &(j, w) in &self.adj[i] {
                let wa = w.abs();
                if w >= 0 {
                    terms.push(format!("(state[{j}] ? 32'sd{wa} : -32'sd{wa})"));
                } else {
                    terms.push(format!("(state[{j}] ? -32'sd{wa} : 32'sd{wa})"));
                }
            }
            v.push_str(&format!("  wire signed [31:0] f{i} = {};\n", terms.join(" + ")));
            v.push_str(&format!(
                "  wire signed [31:0] fc{i} = f{i} > 32'sd{FMAX} ? 32'sd{FMAX} : (f{i} < -32'sd2048 ? -32'sd2048 : f{i});\n"
            ));
            v.push_str(&format!("  wire [9:0] ad{i} = (fc{i} + 32'sd2048) >>> 2;\n"));
            v.push_str(&format!("  wire [31:0] nr{i} = xs32(rng[{i}]);\n"));
            v.push_str(&format!("  wire up{i} = nr{i}[31:16] < fsig(ad{i});\n"));
        }
        v.push_str("\n  always @(posedge clk) begin\n    if (rst) begin\n      phase <= 1'b0;\n");
        for i in 0..n {
            v.push_str(&format!("      rng[{i}] <= 32'd{};\n", self.seeds[i]));
            v.push_str(&format!("      state[{i}] <= 1'b{};\n", self.init_s[i] as u8));
        }
        v.push_str("    end else if (en) begin\n      phase <= ~phase;\n");
        for (ci, class) in self.classes.iter().enumerate() {
            v.push_str(&format!("      if (phase == 1'b{ci}) begin\n"));
            for &iu in class {
                let i = iu as usize;
                v.push_str(&format!("        state[{i}] <= up{i}; rng[{i}] <= nr{i};\n"));
            }
            v.push_str("      end\n");
        }
        v.push_str("    end\n  end\nendmodule\n");
        v
    }

    /// Emit an **AXI4-Lite control shell** around this fabric — the wrapper a real board needs.
    ///
    /// [`Self::emit_verilog`] produces the whole sampler, and it is already portable: the emitted
    /// RTL instantiates **no vendor primitives at all**, so the same text synthesises for an
    /// Artix-7 on a desk, a Zynq `UltraScale+` on a Kria KV260, and an `XCVU47P` on an AWS
    /// `f2.6xlarge`. What a board adds is not a different fabric. It is a way for a host to start
    /// one, stop it, and read it back, and that is the only thing this emits.
    ///
    /// One shell serves both platforms because both speak AXI4-Lite: on F2 it hangs off the
    /// shell's OCL port, and on a KV260 off the PS's `M_AXI_HPM0_FPD`. Only the platform wrapper
    /// and the clock name differ.
    ///
    /// # The register map
    ///
    /// Byte offsets, 32-bit words, `n` the p-bit count:
    ///
    /// | offset | access | meaning |
    /// |---|---|---|
    /// | `0x00` | RW | `bit0` run, `bit1` write-1 to pulse a reset |
    /// | `0x04` | RO | `bit0` running, `bit1` target reached |
    /// | `0x08` | RW | target sweeps; `0` free-runs |
    /// | `0x0C` | RO | sweeps completed |
    /// | `0x10` | RO | `popcount(state)` — magnetisation without reading `n` bits back |
    /// | `0x20 + 4k` | RO | state words, `ceil(n/32)` of them |
    ///
    /// A sweep is two clocks, one per colour class, so the counter advances on the phase that
    /// closes it rather than on every edge.
    ///
    /// `0x10` carries **two cycles of pipeline latency**, and that is bought rather than conceded.
    /// A flat combinational popcount over every state bit was measured as the critical path on an
    /// `xck26`: at 576 p-bits it missed 250 MHz by 0.432 ns with five failing endpoints, and Fmax
    /// fell 447 to 288 to 226 MHz across 64, 256 and 576. Splitting it into per-word counts and
    /// their sum, both registered, recovers **1.968 ns of slack for 33 extra LUTs** — 0.13% area for
    /// 1.80x the clock, and zero failing endpoints. On a register a host polls asynchronously the
    /// latency costs nothing.
    ///
    /// # What this deliberately does not do
    ///
    /// It exposes no power register. Board power is measured by the platform — the AWS shell's own
    /// management path, or the INA rails a KV260 publishes through `hwmon` — never by the custom
    /// logic. A register here that looked like watts would be a modelled number wearing a
    /// measurement's clothes, which is the one thing this crate refuses to ship.
    #[must_use]
    pub fn emit_axi_shell(&self, module: &str, core: &str, clock: &str) -> String {
        let n = self.n;
        let words = n.div_ceil(32);
        let top = n - 1;
        let mut v = String::new();
        v.push_str(&format!(
"// generated by ferrotherm::hdl — AXI4-Lite control shell around `{core}`
// {n} p-bits in {words} state word(s). Portable: AWS F2 (OCL port) and Kria KV260 (HPM0_FPD).
module {module} (
    input  wire        {clock},
    input  wire        rst_n,
    input  wire [31:0] s_axi_awaddr,  input  wire s_axi_awvalid, output wire s_axi_awready,
    input  wire [31:0] s_axi_wdata,   input  wire [3:0] s_axi_wstrb,
    input  wire        s_axi_wvalid,  output wire s_axi_wready,
    output wire [1:0]  s_axi_bresp,   output wire s_axi_bvalid,  input  wire s_axi_bready,
    input  wire [31:0] s_axi_araddr,  input  wire s_axi_arvalid, output wire s_axi_arready,
    output wire [31:0] s_axi_rdata,   output wire [1:0] s_axi_rresp,
    output wire        s_axi_rvalid,  input  wire s_axi_rready
);
  reg soft_rst = 1'b0;
  reg run = 1'b0;
  reg [31:0] target = 32'd0;
  reg [31:0] done_sweeps = 32'd0;
  wire rst = ~rst_n | soft_rst;
  wire reached = (target != 32'd0) && (done_sweeps >= target);
  wire [{top}:0] state;
  wire phase;
  {core} core (.clk({clock}), .rst(rst), .en(run & ~reached), .state(state), .phase(phase));

  // A sweep is two clocks. Count on the closing phase, not on every edge.
  reg phase_d = 1'b0;
  always @(posedge {clock}) begin
    if (rst) begin phase_d <= 1'b0; done_sweeps <= 32'd0; end
    else if (run & ~reached) begin
      phase_d <= phase;
      if (phase_d & ~phase) done_sweeps <= done_sweeps + 32'd1;
    end
  end

  // popcount, so a host reads magnetisation in one word instead of pulling n bits over AXI --
  // PIPELINED, because the flat version WAS the critical path and this is measured, not guessed.
  // One always @(*) summing all {n} bits is an n-input adder tree: on an xck26 at 576 p-bits it
  // missed 250 MHz by 0.432 ns with 5 failing endpoints, and Fmax fell 447 -> 288 -> 226 MHz across
  // 64 -> 256 -> 576. Two registered stages -- per-32-bit-word counts, then their sum -- cost two
  // cycles of latency on a status register a host polls asynchronously, and buy the clock back.
  //
  // The padding is a generate-if rather than a replication because {{0{{1'b0}}}} is illegal Verilog,
  // and n is a multiple of 32 exactly when the fabric is a convenient size.
  localparam NPB = {n};
  localparam NW  = (NPB + 31) / 32;
  wire [NW*32-1:0] pstate;
  genvar gi;
  generate for (gi = 0; gi < NW*32; gi = gi + 1) begin : pad
    if (gi < NPB) begin : real_bit
      assign pstate[gi] = state[gi];
    end else begin : zero_bit
      assign pstate[gi] = 1'b0;
    end
  end endgenerate
  function [6:0] pc32; input [31:0] x; integer k; begin
    pc32 = 7'd0;
    for (k = 0; k < 32; k = k + 1) pc32 = pc32 + {{6'd0, x[k]}};
  end endfunction
  reg [6:0] cpop [0:NW-1];
  reg [31:0] popc;
  integer wi, wj;
  reg [31:0] pacc;
  always @(posedge {clock}) for (wi = 0; wi < NW; wi = wi + 1) cpop[wi] <= pc32(pstate[wi*32 +: 32]);
  always @(posedge {clock}) begin
    pacc = 32'd0;
    for (wj = 0; wj < NW; wj = wj + 1) pacc = pacc + {{25'd0, cpop[wj]}};
    popc <= pacc;
  end

  // AXI4-Lite slave: one outstanding transaction, which is all a register file needs.
  reg awready_r = 1'b0, wready_r = 1'b0, bvalid_r = 1'b0, arready_r = 1'b0, rvalid_r = 1'b0;
  reg aw_seen = 1'b0, w_seen = 1'b0;
  reg [31:0] awaddr_r = 32'd0, araddr_r = 32'd0, rdata_r = 32'd0, wdata_r = 32'd0;
  assign s_axi_awready = awready_r; assign s_axi_wready = wready_r;
  assign s_axi_bvalid  = bvalid_r;  assign s_axi_bresp  = 2'b00;
  assign s_axi_arready = arready_r; assign s_axi_rvalid = rvalid_r;
  assign s_axi_rdata   = rdata_r;   assign s_axi_rresp  = 2'b00;

  always @(posedge {clock}) begin
    if (!rst_n) begin
      awready_r <= 1'b0; wready_r <= 1'b0; bvalid_r <= 1'b0;
      arready_r <= 1'b0; rvalid_r <= 1'b0;
      aw_seen <= 1'b0; w_seen <= 1'b0;
      run <= 1'b0; target <= 32'd0; soft_rst <= 1'b0;
    end else begin
      soft_rst <= 1'b0;
      // AXI4-Lite's write address and write data are INDEPENDENT channels: a master may present
      // them in either order or together, and the write commits only once both have arrived. The
      // first cut of this collapsed them into one `if` and decoded `awaddr_r` in the same cycle it
      // was being loaded, so the case read the PREVIOUS address and every write landed on the wrong
      // register. It also drove `bvalid` from the data beat alone, which meant the response could be
      // consumed before a master that waits for both handshakes ever looked for it.
      if (s_axi_awvalid && !aw_seen) begin
        awaddr_r <= s_axi_awaddr;
        aw_seen <= 1'b1;
        awready_r <= 1'b1;
      end else awready_r <= 1'b0;
      if (s_axi_wvalid && !w_seen) begin
        wdata_r <= s_axi_wdata;
        w_seen <= 1'b1;
        wready_r <= 1'b1;
      end else wready_r <= 1'b0;
      // Commit on the cycle after both beats have landed, so `awaddr_r` and `wdata_r` are settled.
      if (aw_seen && w_seen && !bvalid_r) begin
        case (awaddr_r[7:0])
          8'h00: begin run <= wdata_r[0]; soft_rst <= wdata_r[1]; end
          8'h08: target <= wdata_r;
          default: ;
        endcase
        bvalid_r <= 1'b1;
      end
      if (bvalid_r && s_axi_bready) begin
        bvalid_r <= 1'b0;
        aw_seen <= 1'b0;
        w_seen <= 1'b0;
      end
      if (s_axi_arvalid && !arready_r && !rvalid_r) begin araddr_r <= s_axi_araddr; arready_r <= 1'b1; end
      else arready_r <= 1'b0;
      if (arready_r) begin
        rvalid_r <= 1'b1;
        case (araddr_r[7:0])
          8'h00: rdata_r <= {{30'd0, 1'b0, run}};
          8'h04: rdata_r <= {{30'd0, reached, run & ~reached}};
          8'h08: rdata_r <= target;
          8'h0C: rdata_r <= done_sweeps;
          8'h10: rdata_r <= popc;
"));
        for k in 0..words {
            let lo = k * 32;
            let hi = ((k + 1) * 32 - 1).min(n - 1);
            let expr = if hi - lo + 1 == 32 {
                format!("state[{hi}:{lo}]")
            } else {
                format!("{{{}'d0, state[{hi}:{lo}]}}", 32 - (hi - lo + 1))
            };
            v.push_str(&format!("          8'h{:02X}: rdata_r <= {expr};\n", 0x20 + 4 * k));
        }
        v.push_str(
"          default: rdata_r <= 32'hDEADBEEF;
        endcase
      end else if (rvalid_r && s_axi_rready) rvalid_r <= 1'b0;
    end
  end
endmodule
");
        v
    }

    /// Emit a self-checking testbench plus the expected per-sweep state trace (hex lines) from
    /// the emulator. The testbench prints `FERROTHERM_PASS` only on a bit-exact match.
    pub fn emit_testbench(&mut self, module: &str, sweeps: usize) -> (String, String) {
        self.reset();
        let n = self.n;
        let hexw = n.div_ceil(4);
        let mut expected = String::new();
        for _ in 0..sweeps {
            self.sweep();
            let mut val = vec![0u8; hexw];
            for (i, &b) in self.s.iter().enumerate() {
                if b {
                    val[hexw - 1 - i / 4] |= 1 << (i % 4);
                }
            }
            for b in &val {
                expected.push_str(&format!("{b:x}"));
            }
            expected.push('\n');
        }
        self.reset();
        let tb = format!(
            "`timescale 1ns/1ps\nmodule tb;\n  reg clk = 0, rst = 1;\n  wire [{top}:0] state;\n  wire phase;\n  {module} dut(.clk(clk), .rst(rst), .en(1'b1), .state(state), .phase(phase));\n  reg [{top}:0] expected [0:{last}];\n  integer sw, errors = 0;\n  always #5 clk = ~clk;\n  initial begin\n    $readmemh(\"expected.hex\", expected);\n    @(posedge clk); @(posedge clk); rst = 0;\n    for (sw = 0; sw < {sweeps}; sw = sw + 1) begin\n      @(posedge clk); @(posedge clk); #1;\n      if (state !== expected[sw]) begin\n        errors = errors + 1;\n        $display(\"MISMATCH sweep %0d: got %h want %h\", sw, state, expected[sw]);\n      end\n    end\n    if (errors == 0) $display(\"FERROTHERM_PASS\");\n    else $display(\"FERROTHERM_FAIL %0d\", errors);\n    $finish;\n  end\nendmodule\n",
            top = n - 1,
            last = sweeps - 1,
        );
        (tb, expected)
    }
}


// -- the emitted fabric, behind the same trait as every other backend ---------------------------

/// The p-bit fabric this module emits, as a [`crate::fabric::Device`].
///
/// The gap this closes is the one the roadmap calls *"the same `.ftp` on CPU, browser and FPGA"*.
/// `Cpu` and the GPU backend were both reachable through [`crate::fabric::Device`]; the fabric that
/// actually ran on silicon was reachable only by calling [`FixedFabric`] by hand. So the one
/// backend with a measured joules figure was the one a `.ftp` could not be pointed at.
///
/// # What running here proves, and what it does not
///
/// [`FixedFabric`] is a cycle-exact emulator of the emitted Verilog, and the icarus-verilog gate in
/// this module's tests replays its per-sweep state trace against the RTL and requires a bit-exact
/// match. So a distribution produced here is the distribution the **netlist** produces, not an
/// approximation of it — and that netlist is what was implemented for `xck26` and metered on a
/// Kria KV260 at 10.85 pJ per node update.
///
/// It does **not** prove a board did it. This is silicon semantics without the silicon; the board
/// arm of that claim is the measurement in [`crate::ledger::KV260_MEASURED`], and the two meet at
/// the bit-exactness gate rather than in one process.
///
/// # Why β costs a write here
///
/// The fabric quantises `β·J` into the netlist: the weights and the sigmoid ROM are constants in
/// the emitted Verilog, so a schedule that changes temperature is a schedule that changes the
/// **bitstream**. Every rung after the first is charged `writes += n` for that reason, and on real
/// hardware it is a full reconfiguration rather than a register poke. A twelve-rung ladder is
/// twelve implementations, which is a fact about this fabric worth meeting in the ledger rather
/// than in a synthesis queue.
pub struct RtlFabric {
    graph: Option<Graph>,
    state: Vec<i8>,
    max_spins: Option<usize>,
    /// Whether the bitstream `program` paid for is still the one in the fabric.
    ///
    /// `run` charges a reflash per rung because each rung is a different implementation. It used to
    /// decide that with `if rung > 0` — relative to the CALL, so the first rung of a *second* run
    /// was free even though the fabric was then holding the last run's final configuration and its
    /// seeds. One full-graph reflash is the most expensive line in the ledger (at `Z1_SPICE` a
    /// write is worth 21,664 node updates), so the missing term was the largest one.
    load_unused: bool,
    ledger: crate::ledger::Ledger,
}

impl RtlFabric {
    /// The emulator with no board-size limit: the RTL's semantics, unconstrained by any part.
    #[must_use]
    pub fn new() -> RtlFabric {
        RtlFabric {
            graph: None,
            state: Vec::new(),
            max_spins: None,
            load_unused: false,
            ledger: crate::ledger::Ledger::default(),
        }
    }

    /// The fabric as it fits on one FPGA, sized by [`crate::targets::FpgaTarget::measured_pbits`].
    ///
    /// Measured density, not a model: 44.3 LUTs per p-bit on `xck26`, which is about twice the
    /// generic figure and so admits about half as many p-bits as the generic model promises.
    #[must_use]
    pub fn on(target: &crate::targets::FpgaTarget) -> RtlFabric {
        RtlFabric { max_spins: Some(target.measured_pbits() as usize), ..RtlFabric::new() }
    }

    /// The declared capabilities, without building anything.
    #[must_use]
    pub fn describe(max_spins: Option<usize>) -> crate::fabric::Fabric {
        use crate::fabric::{Precision, Range};
        let mut f = crate::fabric::Fabric::unconstrained("ferrotherm-pbit-rtl", crate::ledger::KV260_MEASURED);
        f.max_spins = max_spins;
        f.max_arity = 2;
        // Q.8 on an ABSOLUTE grid: `FixedFabric::new` computes `(w * 256).round()`, with no
        // reference to the largest coefficient present.
        //
        // This said `Precision::Fixed { bits: 12 }`, which in this crate means something else --
        // a step of `max|w| / (2^(bits-1) - 1)`, i.e. a fabric that NORMALISES. Under that
        // declaration `Fabric::check` waved through a program whose weights were all 0.001,
        // computing a relative error of ~0, and the fabric then quantised every one of them to
        // zero and sampled a graph with no couplings in it. `Precision::Grid` is the variant that
        // says what this hardware actually does, and `check` now refuses that program.
        f.coupling_precision = Precision::Grid { step: 1.0 / (1u32 << FRAC) as f64 };
        f.field_precision = Precision::Grid { step: 1.0 / (1u32 << FRAC) as f64 };
        f.coupling_range = Some(Range::continuous(-8.0, 2047.0 / 256.0));
        f.field_range = Some(Range::continuous(-8.0, 2047.0 / 256.0));
        f
    }
}

impl Default for RtlFabric {
    fn default() -> Self {
        RtlFabric::new()
    }
}

impl crate::fabric::Device for RtlFabric {
    fn fabric(&self) -> crate::fabric::Fabric {
        RtlFabric::describe(self.max_spins)
    }

    fn program(&mut self, p: &crate::ftp::Program) -> Vec<crate::fabric::Unsupported> {
        let mut bad = self.fabric().check(p);
        if !bad.is_empty() {
            return bad;
        }
        match p.to_graph() {
            Ok(g) => {
                // Checked rather than asserted: `FixedFabric::new` PANICS on a graph that is not
                // two-colourable, and a backend that aborts the process is not a backend that can
                // be offered a program. The v1 fabric updates two colour classes and has nowhere
                // to put a third; saying so is the difference between a refusal and a crash.
                if g.classes.len() != 2 {
                    bad.push(crate::fabric::Unsupported::Unplaceable {
                        detail: format!(
                            "this fabric updates two colour classes in two clocks and the program \
                             needs {}; colour it into two, or run it on a backend that schedules \
                             colours dynamically",
                            g.classes.len()
                        ),
                    });
                    return bad;
                }
                self.state = vec![-1; g.n];
                // A load is a write, charged. See `Device::program`. That write buys the first
                // configuration `run` needs, so `run` does not charge again for it -- but it buys
                // exactly one, and every rung after it is another implementation.
                self.ledger.writes += g.n as u64;
                self.load_unused = true;
                self.graph = Some(g);
            }
            Err(e) => bad.push(crate::fabric::Unsupported::Unplaceable { detail: e.to_string() }),
        }
        bad
    }

    fn run(&mut self, schedule: &crate::schedule::Schedule, seed: u64) -> Result<Vec<i8>, String> {
        let Some(g) = self.graph.as_ref() else {
            return Err("no program loaded".into());
        };
        if schedule.stages().is_empty() {
            return Err("a schedule with no stages advances nothing".into());
        }
        // The best state the schedule reached, per the trait -- and the readback that finding it
        // costs, charged. On this fabric that cost is not notional: a p-bit array holds its state
        // on-chip, so scoring a sweep means carrying n spins to the host, and the board this
        // netlist ran on prices a node update at 10.85 pJ with the readback unstated. A ladder
        // that inspects every sweep reads far more than it samples.
        let mut best: Option<Vec<i8>> = None;
        let mut best_e = f64::INFINITY;
        let mut last: Option<Vec<bool>> = None;
        for (rung, stage) in schedule.stages().iter().enumerate() {
            // Each rung is its own quantisation of beta*J, which on hardware is its own bitstream.
            let mut fab = FixedFabric::new(g, stage.beta, seed ^ (rung as u64).wrapping_mul(0x9E37));
            // EACH RUNG STARTS WHERE THE NETLIST STARTS, which is not where the last one finished.
            //
            // This used to copy the previous rung's spins in, so a ladder annealed. The emitted
            // hardware cannot do that: `emit_verilog`'s module has ports `clk, rst, en`, an OUTPUT
            // `state` and `phase` — there is no state input — and the AXI shell's state words at
            // `0x20 + 4k` are read-only. A reconfigured fabric comes up from the seeds baked into
            // its bitstream, full stop. Carrying state made this backend better than the board it
            // models, which is the direction a model must never err in: the doc on this type says
            // a distribution produced here is the NETLIST's, and it has to stay true.
            //
            // So a multi-rung schedule on this fabric is what it is on the hardware: N independent
            // implementations, each run from reset, best answer kept.
            // A new bitstream is a full reprogram of every node, and the ledger says so. Charged
            // against what the FABRIC is holding, not against the position in this loop: after any
            // rung has run, the netlist carries that rung's weights and seeds, so the next
            // configuration -- including the first rung of the next `run` -- is a reflash.
            if self.load_unused {
                self.load_unused = false;
            } else {
                self.ledger.writes += g.n as u64;
            }
            for _ in 0..stage.sweeps {
                fab.sweep();
                let st: Vec<i8> = fab.s.iter().map(|&up| if up { 1i8 } else { -1 }).collect();
                let e = g.energy(&st);
                if e < best_e {
                    best_e = e;
                    best = Some(st);
                }
            }
            self.ledger.samples += (g.n as u64) * (stage.sweeps as u64);
            self.ledger.reads += (g.n as u64) * (stage.sweeps as u64);
            last = Some(fab.s.clone());
        }
        // A schedule whose stages all declare zero sweeps advances nothing, so there is no state
        // to have been best -- fall back to the configuration the fabric came up in rather than
        // panicking on a program that is merely pointless.
        self.state = best.unwrap_or_else(|| {
            last.expect("at least one stage")
                .iter()
                .map(|&up| if up { 1i8 } else { -1 })
                .collect()
        });
        Ok(self.state.clone())
    }

    fn sample(
        &mut self,
        beta: f64,
        plan: &crate::samples::Plan,
        seed: u64,
    ) -> Result<crate::samples::SampleSet, String> {
        let Some(g) = self.graph.as_ref() else {
            return Err("no program loaded".into());
        };
        // One temperature, so one quantisation of beta*J -- which is one bitstream. This is the
        // case the fabric is actually built for; it is the annealing ladder in `run` that costs a
        // reimplementation per rung.
        let mut fab = FixedFabric::new(g, beta, seed);
        for _ in 0..plan.burn_in {
            fab.sweep();
        }
        let thin = plan.thin.max(1);
        let mut states = Vec::with_capacity(plan.draws);
        let mut energies = Vec::with_capacity(plan.draws);
        for _ in 0..plan.draws {
            for _ in 0..thin {
                fab.sweep();
            }
            let st: Vec<i8> = fab.s.iter().map(|&up| if up { 1i8 } else { -1 }).collect();
            energies.push(g.energy(&st));
            states.push(st);
        }
        self.ledger.samples += (g.n as u64) * (plan.sweeps() as u64);
        // Every draw leaves the fabric. On the metered board that is the term worth 239 updates
        // apiece, so a chain of 3,000 draws costs far more in readback than in sampling.
        self.ledger.reads += (plan.draws as u64) * (g.n as u64);
        Ok(crate::samples::SampleSet::from_chain(states, energies, beta, plan.burn_in, thin))
    }

    fn ledger(&self) -> crate::ledger::Ledger {
        self.ledger
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::ising::{lattice2d, onsager_m};

    /// The quantized fabric must still pass the physics gate: Onsager within quantization tolerance.
    #[test]
    fn fixed_point_physics() {
        let g = lattice2d(24, 1.0);
        let beta = 0.5;
        let mut fab = FixedFabric::new(&g, beta, 0xFAB);
        fab.s.fill(true);
        for _ in 0..1500 {
            fab.sweep();
        }
        let mut acc = 0.0;
        let reads = 3000;
        for _ in 0..reads {
            fab.sweep();
            acc += fab.magnetization().abs();
        }
        let m = acc / reads as f64;
        let exact = onsager_m(beta);
        assert!((m - exact).abs() < 0.03, "fixed-point |M| {m:.4} vs Onsager {exact:.4}");
    }


    /// THE SHELL GATE: the AXI4-Lite wrapper is driven like a host would drive it, in simulation,
    /// and must start the fabric, count sweeps, stop at the target, and read back the SAME state
    /// the emulator reaches. A control path that compiles but miscounts a sweep or returns a stale
    /// word is exactly the defect this catches, and it cannot be caught by staring at the RTL.
    #[test]
    fn axi_shell_runs_the_fabric_and_reads_back_what_the_emulator_reaches() {
        if std::process::Command::new("iverilog").arg("-V").output().is_err() {
            eprintln!("SKIP: iverilog not installed; the AXI shell gate did not run");
            return;
        }
        let sweeps = 25usize;
        let g = lattice2d(4, 0.9);
        let mut fab = FixedFabric::new(&g, 0.7, 0x5EED);
        let core = fab.emit_verilog("fabric");
        let shell = fab.emit_axi_shell("ft_axi", "fabric", "clk");
        // What the emulator reaches after exactly `sweeps` sweeps -- the answer the host must read.
        fab.reset();
        for _ in 0..sweeps {
            fab.sweep();
        }
        let want: u32 = fab.s.iter().enumerate().fold(0u32, |a, (i, &b)| a | (u32::from(b) << i));
        let want_pop = fab.s.iter().filter(|&&b| b).count() as u32;

        let tb = format!(
            r#"`timescale 1ns/1ps
module tb;
  reg clk = 0, rst_n = 0;
  reg [31:0] awaddr = 0, wdata = 0, araddr = 0;
  reg awvalid = 0, wvalid = 0, bready = 1, arvalid = 0, rready = 1;
  wire awready, wready, bvalid, arready, rvalid;
  wire [31:0] rdata; wire [1:0] bresp, rresp;
  ft_axi dut(.clk(clk), .rst_n(rst_n),
    .s_axi_awaddr(awaddr), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
    .s_axi_wdata(wdata), .s_axi_wstrb(4'hF), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
    .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
    .s_axi_araddr(araddr), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
    .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid), .s_axi_rready(rready));
  always #5 clk = ~clk;
  reg [31:0] got_state, got_pop, got_done, got_status;
  integer guard;
  // AXI4-Lite's two write channels are independent and their readies need not coincide, so each
  // valid is dropped on its OWN ready. The first cut waited for both at once and then looked for
  // `bvalid` a cycle later -- by which time `bready` being tied high had already consumed the
  // response, and the task waited forever. That hang is what this task's shape is for.
  reg aw_done, w_done;
  task wr(input [31:0] a, input [31:0] d);
    begin
      aw_done = 0; w_done = 0;
      @(posedge clk); awaddr <= a; wdata <= d; awvalid <= 1; wvalid <= 1;
      while (!aw_done || !w_done) begin
        @(posedge clk);
        if (awready) begin awvalid <= 0; aw_done = 1; end
        if (wready)  begin wvalid  <= 0; w_done  = 1; end
      end
      while (!bvalid) @(posedge clk);
      @(posedge clk);
    end
  endtask
  task rd(input [31:0] a, output [31:0] d);
    begin
      @(posedge clk); araddr <= a; arvalid <= 1;
      @(posedge clk); while (!arready) @(posedge clk);
      arvalid <= 0;
      while (!rvalid) @(posedge clk);
      d = rdata;
      @(posedge clk);
    end
  endtask
  // A watchdog, because the failure this gate actually hit was a HANG, not a wrong answer: the
  // suite sat in `vvp` for over an hour and reported nothing. A test that cannot fail in bounded
  // time is not a gate.
  initial begin
    #2000000;
    $display("FERROTHERM_FAIL watchdog: the shell never reached its target");
    $finish;
  end
  initial begin
    repeat (4) @(posedge clk); rst_n = 1; repeat (2) @(posedge clk);
    wr(32'h08, 32'd{sweeps});     // target sweeps
    wr(32'h00, 32'h1);            // run
    guard = 0;
    got_status = 0;
    while ((got_status[1] !== 1'b1) && guard < 100000) begin
      rd(32'h04, got_status); guard = guard + 1;
    end
    if (got_status[1] !== 1'b1) begin $display("FERROTHERM_FAIL never reached target"); $finish; end
    rd(32'h0C, got_done);
    rd(32'h10, got_pop);
    rd(32'h20, got_state);
    if (got_done !== 32'd{sweeps})       $display("FERROTHERM_FAIL sweeps %0d want {sweeps}", got_done);
    else if (got_state !== 32'h{want:08x}) $display("FERROTHERM_FAIL state %h want {want:08x}", got_state);
    else if (got_pop !== 32'd{want_pop}) $display("FERROTHERM_FAIL popcount %0d want {want_pop}", got_pop);
    else $display("FERROTHERM_PASS");
    $finish;
  end
endmodule
"#
        );
        let dir = std::env::temp_dir().join(format!("ferrotherm_axi_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("fabric.v"), core).unwrap();
        std::fs::write(dir.join("shell.v"), shell).unwrap();
        std::fs::write(dir.join("tb.v"), tb).unwrap();
        let out = std::process::Command::new("iverilog")
            .current_dir(&dir)
            .args(["-g2012", "-o", "sim", "fabric.v", "shell.v", "tb.v"])
            .output()
            .unwrap();
        assert!(out.status.success(), "iverilog: {}", String::from_utf8_lossy(&out.stderr));
        let run = std::process::Command::new("vvp").current_dir(&dir).arg("sim").output().unwrap();
        let stdout = String::from_utf8_lossy(&run.stdout);
        assert!(stdout.contains("FERROTHERM_PASS"), "AXI shell gate:\n{stdout}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// THE HARDWARE GATE: the emitted Verilog, simulated with icarus-verilog, must reproduce the
    /// emulator's state trace bit-exactly for every sweep. Skips (with a notice) if iverilog is
    /// not installed; CI installs it.
    #[test]
    fn verilog_matches_emulator_bit_exact() {
        if std::process::Command::new("iverilog").arg("-V").output().is_err() {
            eprintln!("SKIP: iverilog not installed; the RTL bit-exactness gate did not run");
            return;
        }
        let g = lattice2d(6, 0.9);
        let mut fab = FixedFabric::new(&g, 0.7, 0x1234);
        let rtl = fab.emit_verilog("fabric");
        let (tb, expected) = fab.emit_testbench("fabric", 40);
        let dir = std::env::temp_dir().join(format!("ferrotherm_hdl_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("fabric.v"), rtl).unwrap();
        std::fs::write(dir.join("tb.v"), tb).unwrap();
        std::fs::write(dir.join("expected.hex"), expected).unwrap();
        let out = std::process::Command::new("iverilog")
            .current_dir(&dir)
            .args(["-g2012", "-o", "sim", "fabric.v", "tb.v"])
            .output()
            .unwrap();
        assert!(out.status.success(), "iverilog: {}", String::from_utf8_lossy(&out.stderr));
        let run = std::process::Command::new("vvp").current_dir(&dir).arg("sim").output().unwrap();
        let stdout = String::from_utf8_lossy(&run.stdout);
        assert!(stdout.contains("FERROTHERM_PASS"), "RTL/emulator divergence:\n{stdout}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod declared_precision {
    use super::*;
    use crate::fabric::{Device, Precision, Unsupported};
    use crate::ftp::Program;

    /// The fabric must declare the quantisation it performs, not one that flatters it.
    ///
    /// `describe` said `Precision::Fixed { bits: 12 }`. In this crate that means a step of
    /// `max|w| / (2^(bits-1) - 1)` — a fabric that spends its bits on whatever scale it is given.
    /// `FixedFabric::new` does `(w * 256).round()`: an absolute grid. Under the wrong declaration
    /// `Fabric::check` computed a relative error of ~0 for a program whose weights were all 0.001
    /// and accepted it; the fabric then rounded every coupling to zero and sampled a graph with no
    /// edges. Nothing raised, nothing logged, and the answer looked like an answer.
    #[test]
    fn a_program_that_quantises_to_nothing_is_refused_rather_than_run() {
        let mut src = String::from("ftp 1\nname tiny-weights\nspins 4\n");
        for i in 0..4 {
            src.push_str(&format!("factor 0.001 {i} {}\n", (i + 1) % 4));
        }
        let p = Program::from_ftp(&src).expect("a well-formed program");

        // The weights really do vanish on this fabric's grid: 0.001 * 256 = 0.256, which rounds
        // to 0. This is the fact the declaration has to be about.
        assert_eq!((0.001_f64 * 256.0).round() as i32, 0);

        let mut d = RtlFabric::new();
        let bad = d.program(&p);
        let found = bad
            .iter()
            .find(|u| matches!(u, Unsupported::CouplingPrecision { .. }))
            .expect("a program that quantises to an empty graph must be refused");
        match found {
            Unsupported::CouplingPrecision { worst_relative_error, .. } => assert!(
                (*worst_relative_error - 1.0).abs() < 1e-12,
                "a coefficient rounded to zero has lost ALL of itself: {worst_relative_error}"
            ),
            _ => unreachable!(),
        }
    }

    /// A second run reflashes as much as the first, because the fabric is not holding its state.
    ///
    /// `run` charged the reflash with `if rung > 0`, which is relative to the CALL. On a second
    /// run the fabric holds the previous run's final configuration and seeds, so its rung 0 needs
    /// a new bitstream like every other rung — and was free. A full-graph reflash is the most
    /// expensive line in this ledger.
    #[test]
    fn a_second_run_pays_for_its_own_first_bitstream() {
        use crate::schedule::Schedule;

        let mut src = String::from("ftp 1\nname ring\nspins 8\n");
        for i in 0..8 {
            src.push_str(&format!("factor 1 {i} {}\n", (i + 1) % 8));
        }
        let p = Program::from_ftp(&src).expect("a well-formed program");
        let mut d = RtlFabric::new();
        assert!(d.program(&p).is_empty());

        let sched = Schedule::geometric(0.1, 4.0, 4, 5);
        let rungs = sched.stages().len() as u64;

        let after_load = Device::ledger(&d).writes;
        d.run(&sched, 1).unwrap();
        let first = Device::ledger(&d).writes - after_load;
        d.run(&sched, 2).unwrap();
        let second = Device::ledger(&d).writes - after_load - first;

        // The load bought one configuration, so the first run reflashes for the OTHER rungs.
        assert_eq!(first, (rungs - 1) * 8, "first run: {first} writes over {rungs} rungs");
        // The second run holds nothing it can reuse, so it pays for every rung.
        assert_eq!(second, rungs * 8, "second run: {second} writes over {rungs} rungs");
    }

    /// And the declaration names the grid the emitter actually uses.
    #[test]
    fn the_declared_step_is_the_one_the_emitter_quantises_on() {
        let Precision::Grid { step } = RtlFabric::describe(None).coupling_precision else {
            panic!("this fabric quantises to an absolute grid and must say so");
        };
        assert_eq!(step, 1.0 / 256.0, "FRAC is 8, so one unit is 1/256");
        // Not a restatement of the constant: this is what `FixedFabric::new` does to a weight.
        let w = 0.7_f64;
        assert_eq!((w * 256.0).round() as i32, (w / step).round() as i32);
    }

    /// A program at a scale the fabric can hold is still accepted.
    #[test]
    fn an_ordinary_program_is_not_refused_by_the_stricter_declaration() {
        let mut src = String::from("ftp 1\nname ordinary\nspins 4\n");
        for i in 0..4 {
            src.push_str(&format!("factor 1 {i} {}\n", (i + 1) % 4));
        }
        let p = Program::from_ftp(&src).expect("a well-formed program");
        let mut d = RtlFabric::new();
        assert!(d.program(&p).is_empty(), "unit couplings sit exactly on a 1/256 grid");
    }
}

#[cfg(test)]
mod kernel_agreement {
    /// The RTL threshold table is a hardware copy of the software update. Hardware cannot call
    /// `kernel::p_up`, so the only thing standing between the emitted Verilog and a silently
    /// different distribution is this test.
    #[test]
    fn lut_agrees_with_the_kernel() {
        const LUT_BITS: usize = 10;
        for &beta in &[0.25, 0.5, 1.0, 2.0, 4.0] {
            let scale = 256.0;
            for a in 0..(1usize << LUT_BITS) {
                let arg = ((a as f64 + 0.5) * 4.0 - 2048.0) / scale;
                let want = crate::kernel::p_up(arg, beta);
                let quantized = (want * 65535.0).round().min(65535.0) as u16;
                // the table is 16-bit; agreement means within one least significant bit
                let back = quantized as f64 / 65535.0;
                assert!(
                    (back - want).abs() <= 1.0 / 65535.0,
                    "beta {beta} entry {a}: table {back} vs kernel {want}"
                );
            }
        }
    }
}
