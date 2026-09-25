//! A p-bit fabric whose couplings are REGISTERS, so a new problem is a write and not a bitstream.
//!
//! [`crate::hdl::FixedFabric`] bakes every weight into the netlist. That is why it is small — a
//! node's field takes a handful of values, so its sigmoid ROM constant-folds to almost nothing —
//! and it is also why [`crate::hdl::RtlFabric`] charges a full reconfiguration for every rung of a
//! temperature ladder, and why the board that metered a flip and a read could not meter a write:
//! there was nothing to write to.
//!
//! This is the same sampler with the constants turned into state. The wiring is still fixed at
//! synthesis (which node neighbours which); what each wire WEIGHS, and each node's bias, live in
//! 12-bit signed registers a host writes over the bus.
//!
//! # The contract
//!
//! **A writable fabric programmed with a graph IS the fixed fabric for that graph, state for
//! state.** Same Q.8 quantisation, same ROM, same per-node `xorshift32`, same chromatic schedule —
//! so [`WritableFabric`] does not carry its own emulator. It carries a [`FixedFabric`] and mutates
//! its weights, and the icarus-verilog gate holds the emitted RTL, after host writes, to the trace
//! of a `FixedFabric` built directly from the target graph. Two fabrics, one trajectory, or the
//! build fails.
//!
//! # What it costs, and why the schedule changes
//!
//! With arbitrary weights the field is arbitrary, the ROM address is arbitrary, and the ROM can no
//! longer fold: every node needs all 1,024 entries. In lookup tables that is ~340 LUT6 per node,
//! more than the rest of the node three times over, so the ROM goes into block RAM — one
//! dual-port 18 kb block serves two nodes — and block RAM reads are REGISTERED. A colour class
//! therefore takes two clocks (present the address; compare and update) and a sweep takes four,
//! against the fixed fabric's two. The state sequence per sweep is unchanged, because a class
//! reads only the other class, which holds still for both clocks. Half the sweep rate is the
//! price of being programmable on this part, and it is stated rather than hidden in a clock.
//!
//! # Temperature is a write too
//!
//! `β` lives in the ROM, which is not writable. But the ROM is indexed by the FIELD, and the field
//! is linear in the weights, so writing `c·w` and `c·h` samples at `c·β`. A ladder of temperatures
//! on this fabric is a ladder of register writes, inside the ±8.0 range of Q.8 in twelve bits —
//! which [`WritableFabric::program`] refuses to exceed rather than clamping a weight silently.
//!
//! # On silicon: what was seen, and what was not
//!
//! Built for a Kria KV260 on 2026-09-19 (`examples/board_build -- kv260-axi-w`, 256 p-bits):
//! 35,152 LUTs and 24,808 registers — **137 LUTs a p-bit against the fixed fabric's 44** — 128
//! `RAMB18`, exactly one per pair of nodes, and timing met at 100 MHz with 0.550 ns to spare.
//! Loaded once, for a 40-second smoke run:
//!
//! - every bias written to `+max` drove the popcount to 256 of 256, and to `-max`, to 0;
//! - the shell accepted 2,304 configuration words of 2,304 sent, then 68,241,408 of 68,241,408;
//! - 24,999,53x sweeps a second against a known answer of `2.5e7`, and zero when held;
//! - one A53 core wrote 7.58 M words a second — 2.53 M node configurations a second.
//!
//! **A write was NOT metered.** That run's power samples were not kept, and within a minute of
//! its last line the board stopped answering SSH (it still answered ping), before the eight-pass
//! protocol could start. The cause is not known. The one hypothesis that could be tested without
//! the board — that this shell deadlocks under overlapping or stalled bus transactions — was
//! tested in simulation and NOT supported: see
//! `neither_shell_deadlocks_under_a_badly_behaved_master`. What that gate does not cover is an
//! interconnect's behaviour across a reconfiguration, which is where suspicion now sits. `e_write`
//! stays unstated in every price set this crate owns.

use crate::graph::Graph;
use crate::hdl::{FixedFabric, FRAC, LUT_BITS};

/// Bits in a weight or bias register: signed Q.8, so `[-8.0, +8.0)`.
pub const WEIGHT_BITS: u32 = 12;
const WMIN: i32 = -(1 << (WEIGHT_BITS - 1));
const WMAX: i32 = (1 << (WEIGHT_BITS - 1)) - 1;

/// Byte offset of node 0's first configuration word in the AXI shell's address space.
pub const CONFIG_BASE: u32 = 0x1000;

/// Why a graph cannot be written into this fabric.
#[derive(Debug, Clone, PartialEq)]
pub enum WriteError {
    /// The graph's wiring is not the wiring that was synthesised. Weights are writable; wires are
    /// not.
    Topology {
        /// The first node whose neighbour list differs.
        node: usize,
    },
    /// A weight or bias does not fit twelve signed bits of Q.8. Refused, not clamped: a clamped
    /// coupling samples a different problem and reports nothing.
    Range {
        /// The node whose register would overflow.
        node: usize,
        /// The offending value in graph units.
        value: f64,
    },
}

impl std::fmt::Display for WriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WriteError::Topology { node } => {
                write!(f, "node {node} is wired differently from the synthesised fabric")
            }
            WriteError::Range { node, value } => {
                write!(f, "node {node}: {value} does not fit signed Q.8 in {WEIGHT_BITS} bits")
            }
        }
    }
}

impl std::error::Error for WriteError {}

/// One 32-bit bus write: a byte offset in the shell's address space, and the word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BusWrite {
    /// Byte offset from the shell's base address.
    pub offset: u32,
    /// The word written.
    pub data: u32,
}

/// The fixed fabric with its constants turned into registers.
pub struct WritableFabric {
    /// The emulator, and the power-on contents of every register.
    pub core: FixedFabric,
    /// Weight slots per node: the largest degree in the synthesised wiring.
    pub degree: usize,
}

fn quantise(node: usize, value: f64) -> Result<i32, WriteError> {
    let q = (value * f64::from(1u32 << FRAC)).round();
    if !(f64::from(WMIN)..=f64::from(WMAX)).contains(&q) {
        return Err(WriteError::Range { node, value });
    }
    Ok(q as i32)
}

impl WritableFabric {
    /// Synthesise the wiring of `g`, with `g`'s own weights as the power-on register contents.
    ///
    /// # Errors
    ///
    /// [`WriteError::Range`] if a weight or bias of `g` does not fit a register.
    ///
    /// # Panics
    ///
    /// As [`FixedFabric::new`]: if `g` is not bipartite.
    pub fn new(g: &Graph, beta: f64, seed: u64) -> Result<WritableFabric, WriteError> {
        for i in 0..g.n {
            quantise(i, g.h[i])?;
            for k in g.offset[i]..g.offset[i + 1] {
                quantise(i, g.w[k])?;
            }
        }
        let core = FixedFabric::new(g, beta, seed);
        let degree = core.adj.iter().map(Vec::len).max().unwrap_or(0).max(1);
        Ok(WritableFabric { core, degree })
    }

    /// Configuration words per node: two weights to a word, then the bias.
    #[must_use]
    pub fn words_per_node(&self) -> usize {
        self.degree.div_ceil(2) + 1
    }

    /// Bytes of address space a node's configuration occupies (a power of two, for the decoder).
    #[must_use]
    pub fn node_stride(&self) -> u32 {
        (self.words_per_node() * 4).next_power_of_two() as u32
    }

    /// Write `g`'s weights and biases into the fabric, and return the bus writes that do the same
    /// to the hardware — the emulator and the host program are updated by ONE function, so they
    /// cannot describe different problems.
    ///
    /// # Errors
    ///
    /// [`WriteError::Topology`] if `g` is wired differently from the synthesised fabric, and
    /// [`WriteError::Range`] if a value does not fit. On an error nothing has been written.
    pub fn program(&mut self, g: &Graph) -> Result<Vec<BusWrite>, WriteError> {
        if g.n != self.core.n {
            return Err(WriteError::Topology { node: g.n.min(self.core.n) });
        }
        // Validate everything before touching anything.
        let mut staged: Vec<(Vec<i32>, i32)> = Vec::with_capacity(g.n);
        for i in 0..g.n {
            let nbrs = &g.nbr[g.offset[i]..g.offset[i + 1]];
            let wired: Vec<u32> = self.core.adj[i].iter().map(|&(j, _)| j).collect();
            if nbrs != wired.as_slice() {
                return Err(WriteError::Topology { node: i });
            }
            let ws = (g.offset[i]..g.offset[i + 1])
                .map(|k| quantise(i, g.w[k]))
                .collect::<Result<Vec<_>, _>>()?;
            staged.push((ws, quantise(i, g.h[i])?));
        }
        let mut bus = Vec::with_capacity(g.n * self.words_per_node());
        let mask = (1u32 << WEIGHT_BITS) - 1;
        for (i, (ws, bias)) in staged.into_iter().enumerate() {
            let base = CONFIG_BASE + i as u32 * self.node_stride();
            for (word, pair) in (0..self.degree).step_by(2).enumerate() {
                let lo = ws.get(pair).copied().unwrap_or(0) as u32 & mask;
                let hi = ws.get(pair + 1).copied().unwrap_or(0) as u32 & mask;
                bus.push(BusWrite { offset: base + 4 * word as u32, data: lo | (hi << 16) });
            }
            bus.push(BusWrite {
                offset: base + 4 * self.degree.div_ceil(2) as u32,
                data: bias as u32 & mask,
            });
            for (slot, w) in ws.into_iter().enumerate() {
                self.core.adj[i][slot].1 = w;
            }
            self.core.bias_q[i] = bias;
        }
        Ok(bus)
    }

    /// Emit the synthesizable writable fabric.
    ///
    /// Ports beyond the fixed fabric's: `wr_en`, `wr_node`, `wr_word`, `wr_data` — one
    /// configuration word of one node per clock. `rst` restores spins and generators and leaves
    /// the weights alone, so a host can program a problem and then restart the chain on it.
    #[must_use]
    pub fn emit_verilog(&self, module: &str) -> String {
        let n = self.core.n;
        let d = self.degree;
        let nb = usize::BITS - (n - 1).max(1).leading_zeros();
        let wt = WEIGHT_BITS - 1;
        let mut v = String::new();
        v.push_str(&format!(
            "// generated by ferrotherm::writable — p-bit fabric with WRITABLE couplings\n\
             // {n} p-bits, degree {d}, signed Q.{FRAC} in {WEIGHT_BITS} bits, {}-entry sigmoid ROM in block RAM,\n\
             // xorshift32 per node. Four clocks a sweep: a registered ROM read, then the update, per class.\n",
            1usize << LUT_BITS
        ));
        // The ROM, once, as a module: two registered read ports, which is one 18 kb block.
        v.push_str(&format!(
            "module {module}_rom (input wire clk, input wire en,\n    input wire [9:0] a0, input wire [9:0] a1,\n    output reg [15:0] q0, output reg [15:0] q1);\n  (* rom_style = \"block\" *) reg [15:0] rom [0:1023];\n  initial begin\n"
        ));
        for (a, p) in self.core.lut.iter().enumerate() {
            v.push_str(&format!("    rom[{a}] = 16'd{p};\n"));
        }
        v.push_str(
            "  end\n  always @(posedge clk) if (en) begin q0 <= rom[a0]; q1 <= rom[a1]; end\nendmodule\n\n",
        );
        v.push_str(&format!(
            "module {module} (\n    input wire clk,\n    input wire rst,\n    input wire en,\n    input wire wr_en,\n    input wire [{nbt}:0] wr_node,\n    input wire [2:0] wr_word,\n    input wire [31:0] wr_data,\n    output reg [{top}:0] state,\n    output reg phase,\n    output reg sub\n);\n",
            nbt = nb - 1,
            top = n - 1
        ));
        v.push_str(
            "  function [31:0] xs32; input [31:0] x; reg [31:0] a, b; begin\n    a = x ^ (x << 13); b = a ^ (a >> 17); xs32 = b ^ (b << 5);\n  end endfunction\n\n",
        );
        v.push_str(&format!("  reg [31:0] rng [0:{}];\n", n - 1));
        v.push_str(&format!("  reg signed [{wt}:0] wj [0:{}];\n", n * d - 1));
        v.push_str(&format!("  reg signed [{wt}:0] wb [0:{}];\n", n - 1));
        // Power-on register contents: the graph this was synthesised from.
        v.push_str("  initial begin\n");
        let lit = |q: i32| {
            if q < 0 {
                format!("-{WEIGHT_BITS}'sd{}", -q)
            } else {
                format!("{WEIGHT_BITS}'sd{q}")
            }
        };
        for i in 0..n {
            for slot in 0..d {
                let q = self.core.adj[i].get(slot).map_or(0, |&(_, w)| w);
                v.push_str(&format!("    wj[{}] = {};\n", i * d + slot, lit(q)));
            }
            v.push_str(&format!("    wb[{i}] = {};\n", lit(self.core.bias_q[i])));
        }
        v.push_str("  end\n\n");
        // The write port. Two weights to a word, then the bias.
        v.push_str("  always @(posedge clk) if (wr_en) begin\n    case (wr_word)\n");
        for word in 0..d.div_ceil(2) {
            v.push_str(&format!("      3'd{word}: begin\n"));
            v.push_str(&format!(
                "        wj[wr_node * {d} + {}] <= wr_data[{wt}:0];\n",
                2 * word
            ));
            if 2 * word + 1 < d {
                v.push_str(&format!(
                    "        wj[wr_node * {d} + {}] <= wr_data[{}:16];\n",
                    2 * word + 1,
                    16 + wt
                ));
            }
            v.push_str("      end\n");
        }
        v.push_str(&format!(
            "      3'd{}: wb[wr_node] <= wr_data[{wt}:0];\n      default: ;\n    endcase\n  end\n\n",
            d.div_ceil(2)
        ));
        for i in 0..n {
            let mut terms = vec![format!("{{{{4{{wb[{i}][{wt}]}}}}, wb[{i}]}}")];
            for (slot, &(j, _)) in self.core.adj[i].iter().enumerate() {
                let r = i * d + slot;
                terms.push(format!(
                    "(state[{j}] ? {{{{4{{wj[{r}][{wt}]}}}}, wj[{r}]}} : -{{{{4{{wj[{r}][{wt}]}}}}, wj[{r}]}})"
                ));
            }
            v.push_str(&format!("  wire signed [15:0] f{i} = {};\n", terms.join(" + ")));
            v.push_str(&format!(
                "  wire signed [15:0] fc{i} = f{i} > 16'sd2047 ? 16'sd2047 : (f{i} < -16'sd2048 ? -16'sd2048 : f{i});\n"
            ));
            v.push_str(&format!("  wire [15:0] of{i} = fc{i} + 16'sd2048;\n"));
            v.push_str(&format!("  wire [9:0] ad{i} = of{i}[11:2];\n"));
            v.push_str(&format!("  wire [15:0] p{i};\n"));
            v.push_str(&format!("  wire [31:0] nr{i} = xs32(rng[{i}]);\n"));
            v.push_str(&format!("  wire up{i} = nr{i}[31:16] < p{i};\n"));
        }
        // One ROM block per pair of nodes; an odd node out shares its block with itself.
        for pair in (0..n).step_by(2) {
            let b = (pair + 1).min(n - 1);
            let q1 = if b == pair { String::new() } else { format!("p{b}") };
            v.push_str(&format!(
                "  {module}_rom rom{pair} (.clk(clk), .en(en & ~sub), .a0(ad{pair}), .a1(ad{b}), .q0(p{pair}), .q1({q1}));\n"
            ));
        }
        v.push_str(
            "\n  always @(posedge clk) begin\n    if (rst) begin\n      phase <= 1'b0;\n      sub <= 1'b0;\n",
        );
        for i in 0..n {
            v.push_str(&format!("      rng[{i}] <= 32'd{};\n", self.core.seeds[i]));
            v.push_str(&format!("      state[{i}] <= 1'b{};\n", u8::from(self.core.init_s[i])));
        }
        // sub = 0: the ROM latches this class's probabilities. sub = 1: compare, update, move on.
        v.push_str("    end else if (en) begin\n      sub <= ~sub;\n      if (sub) begin\n        phase <= ~phase;\n");
        for (ci, class) in self.core.classes.iter().enumerate() {
            v.push_str(&format!("        if (phase == 1'b{ci}) begin\n"));
            for &iu in class {
                let i = iu as usize;
                v.push_str(&format!("          state[{i}] <= up{i}; rng[{i}] <= nr{i};\n"));
            }
            v.push_str("        end\n");
        }
        v.push_str("      end\n    end\n  end\nendmodule\n");
        v
    }

    /// Emit the AXI4-Lite shell: [`FixedFabric::emit_axi_shell`]'s register map, unchanged, plus
    /// the configuration window at [`CONFIG_BASE`].
    ///
    /// | offset | access | meaning |
    /// |---|---|---|
    /// | `0x00`–`0x10` | | as the fixed shell: run/reset, status, target, sweeps done, popcount |
    /// | `0x14` | RO | configuration words accepted since reset of the shell — the write's own receipt |
    /// | `0x20 + 4k` | RO | state words |
    /// | `CONFIG_BASE + node·stride + 4·word` | WO | two weights to a word, then the bias |
    ///
    /// Configuration is write-only. Reading it back would cost a multiplexer over every register
    /// in the fabric, and there is a better proof that a write landed: the physics changes. Write
    /// every bias to `+max` and the popcount goes to `n`.
    ///
    /// # Panics
    ///
    /// If the fabric has more nodes than the sixteen-bit configuration window addresses (3,840 at
    /// degree four). A larger fabric needs a wider decoder, not a silent wrap.
    #[must_use]
    pub fn emit_axi_shell(&self, module: &str, core: &str, clock: &str) -> String {
        let n = self.core.n;
        let words = n.div_ceil(32);
        let nb = usize::BITS - (n - 1).max(1).leading_zeros();
        let stride_bits = self.node_stride().trailing_zeros();
        assert!(
            CONFIG_BASE as usize + n * self.node_stride() as usize <= 0x1_0000,
            "the configuration window is decoded in 16 address bits and {n} nodes do not fit it"
        );
        let mut v = format!(
            "// generated by ferrotherm::writable — AXI4-Lite shell around writable fabric `{core}`
module {module} (
    input  wire {clock},
    input  wire rst_n,
    input  wire [31:0] s_axi_awaddr,  input  wire s_axi_awvalid, output wire s_axi_awready,
    input  wire [31:0] s_axi_wdata,   input  wire [3:0] s_axi_wstrb,
    input  wire        s_axi_wvalid,  output wire s_axi_wready,
    output wire [1:0]  s_axi_bresp,   output wire s_axi_bvalid,  input  wire s_axi_bready,
    input  wire [31:0] s_axi_araddr,  input  wire s_axi_arvalid, output wire s_axi_arready,
    output wire [31:0] s_axi_rdata,   output wire [1:0] s_axi_rresp,
    output wire        s_axi_rvalid,  input  wire s_axi_rready
);
  reg run = 1'b0;
  reg soft_rst = 1'b0;
  reg [31:0] target = 32'd0;
  reg [31:0] done_sweeps = 32'd0;
  reg [31:0] cfg_writes = 32'd0;
  wire rst = ~rst_n | soft_rst;
  wire reached = (target != 32'd0) && (done_sweeps >= target);
  wire [{top}:0] state;
  wire phase, sub;
  reg cfg_en = 1'b0;
  reg [{nbt}:0] cfg_node = 0;
  reg [2:0] cfg_word = 3'd0;
  reg [31:0] cfg_data = 32'd0;
  {core} core (.clk({clock}), .rst(rst), .en(run & ~reached),
    .wr_en(cfg_en), .wr_node(cfg_node), .wr_word(cfg_word), .wr_data(cfg_data),
    .state(state), .phase(phase), .sub(sub));

  // A sweep is FOUR clocks here: count on the clock that updates the second class.
  always @(posedge {clock}) begin
    if (rst) done_sweeps <= 32'd0;
    else if (run & ~reached) begin
      if (phase & sub) done_sweeps <= done_sweeps + 32'd1;
    end
  end

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
  integer wi, wj_;
  reg [31:0] pacc;
  always @(posedge {clock}) for (wi = 0; wi < NW; wi = wi + 1) cpop[wi] <= pc32(pstate[wi*32 +: 32]);
  always @(posedge {clock}) begin
    pacc = 32'd0;
    for (wj_ = 0; wj_ < NW; wj_ = wj_ + 1) pacc = pacc + {{25'd0, cpop[wj_]}};
    popc <= pacc;
  end

  reg awready_r = 1'b0, wready_r = 1'b0, bvalid_r = 1'b0, arready_r = 1'b0, rvalid_r = 1'b0;
  reg aw_seen = 1'b0, w_seen = 1'b0;
  reg [31:0] awaddr_r = 32'd0, wdata_r = 32'd0, araddr_r = 32'd0, rdata_r = 32'd0;
  assign s_axi_awready = awready_r; assign s_axi_wready = wready_r;
  assign s_axi_bvalid  = bvalid_r;  assign s_axi_bresp  = 2'b00;
  assign s_axi_arready = arready_r; assign s_axi_rvalid = rvalid_r;
  assign s_axi_rdata   = rdata_r;   assign s_axi_rresp  = 2'b00;

  always @(posedge {clock}) begin
    if (!rst_n) begin
      awready_r <= 1'b0; wready_r <= 1'b0; bvalid_r <= 1'b0; arready_r <= 1'b0; rvalid_r <= 1'b0;
      aw_seen <= 1'b0; w_seen <= 1'b0; run <= 1'b0; soft_rst <= 1'b0; target <= 32'd0;
      cfg_en <= 1'b0; cfg_writes <= 32'd0;
    end else begin
      soft_rst <= 1'b0;
      cfg_en <= 1'b0;
      awready_r <= 1'b0; wready_r <= 1'b0;
      // The two write channels are independent: latch each on its own handshake.
      if (s_axi_awvalid && !aw_seen) begin
        awaddr_r <= s_axi_awaddr;
        awready_r <= 1'b1; aw_seen <= 1'b1;
      end
      if (s_axi_wvalid && !w_seen) begin
        wdata_r <= s_axi_wdata;
        wready_r <= 1'b1; w_seen <= 1'b1;
      end
      if (aw_seen && w_seen && !bvalid_r) begin
        if (awaddr_r[15:12] != 4'd0) begin
          // The configuration window: one node's word, handed to the fabric for one clock.
          cfg_en   <= 1'b1;
          cfg_node <= (awaddr_r[15:0] - 16'h{base:04X}) >> {stride_bits};
          cfg_word <= awaddr_r[{word_hi}:2];
          cfg_data <= wdata_r;
          cfg_writes <= cfg_writes + 32'd1;
        end else begin
          case (awaddr_r[7:0])
            8'h00: begin run <= wdata_r[0]; soft_rst <= wdata_r[1]; end
            8'h08: target <= wdata_r;
            default: ;
          endcase
        end
        bvalid_r <= 1'b1; aw_seen <= 1'b0; w_seen <= 1'b0;
      end
      if (bvalid_r && s_axi_bready) begin
        bvalid_r <= 1'b0;
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
          8'h14: rdata_r <= cfg_writes;
",
            top = n - 1,
            nbt = nb - 1,
            base = CONFIG_BASE,
            word_hi = stride_bits - 1,
        );
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
",
        );
        v
    }
}

// -- the writable fabric, behind the same trait as every other backend ---------------------------

/// Temperature the sigmoid ROM is built at. Registers hold `(β / ROM_BETA) · J`, so at the default
/// of one a register holds `β·J` — the dimensionless coupling itself, in Q.8, inside ±8.
pub const ROM_BETA: f64 = 1.0;

/// The writable fabric as a [`crate::fabric::Device`]: **one implementation, and a ladder that
/// anneals.**
///
/// # What actually changes against [`crate::hdl::RtlFabric`], and what does not
///
/// **The count of writes does not fall. It is `n` HIGHER.** A rung at a new temperature rewrites
/// every node's weights, which is `writes += n` — what `RtlFabric` charges for a rung too. But
/// `RtlFabric`'s load IS its first rung's bitstream, while this fabric's load holds the program at
/// [`ROM_BETA`] and the first rung is a further write: `n·(R+1)` against `n·R` for an `R`-rung
/// ladder. An earlier note of mine said this backend would "stop charging a reflash per rung".
/// That was wrong in the ledger's own unit, and a test below pins both counts so it cannot come
/// back as a claim.
///
/// What changes is what a write IS, and what survives one:
///
/// - **State survives.** `RtlFabric` restarts every rung from the seeds in its bitstream, because
///   a reconfigured part comes up from reset and the fixed netlist has no state input: its ladder
///   is N independent runs, best kept. Here a write touches weight registers and nothing else, so
///   spins and generators carry from rung to rung and the ladder is simulated annealing. The
///   icarus-verilog gate reprograms the emitted RTL mid-run, without a reset, and holds it to
///   this emulator bit for bit.
/// - **A write is a bus transaction, not an implementation.** On the board a rung of the fixed
///   fabric is a Vivado run (about seven minutes for the 1,024-p-bit build) and a bitstream load;
///   a rung here was measured at 2.53 M node configurations a second from one A53 core, so 256
///   nodes reprogram in about 100 µs. Neither figure is an energy, and no price set in this crate
///   states `e_write`: this device declares [`Prices::UNSTATED`], because this netlist ran on a
///   board and was never metered.
///
/// # What it refuses
///
/// A register holds `(β / ROM_BETA) · J` in twelve signed bits of Q.8. A rung whose scaled weight
/// leaves ±8, or whose NONZERO weight rounds to zero, is refused by name rather than run as a
/// different problem. And the generators' seeds are reset constants of the netlist, so a `run`
/// with a new seed is a new implementation and is charged as a load.
///
/// [`Prices::UNSTATED`]: crate::ledger::Prices::UNSTATED
pub struct WritableRtl {
    graph: Option<Graph>,
    fabric: Option<WritableFabric>,
    /// The seed the synthesised netlist carries, and whether the load `program` paid for is still
    /// unspent.
    seed: Option<u64>,
    load_unused: bool,
    /// The scale the registers currently hold, so an unchanged temperature is not charged twice.
    held_scale: Option<f64>,
    state: Vec<i8>,
    max_spins: Option<usize>,
    ledger: crate::ledger::Ledger,
}

impl WritableRtl {
    /// The emulator with no board-size limit.
    #[must_use]
    pub fn new() -> WritableRtl {
        WritableRtl {
            graph: None,
            fabric: None,
            seed: None,
            load_unused: false,
            held_scale: None,
            state: Vec::new(),
            max_spins: None,
            ledger: crate::ledger::Ledger::default(),
        }
    }

    /// `g` with every weight and bias multiplied by `c`, or the reason a register cannot hold it.
    fn scaled(g: &Graph, c: f64) -> Result<Graph, String> {
        // `Graph` is deliberately not `Clone`; every field is public, so a scaled copy is spelled out.
        let mut out = Graph {
            n: g.n,
            offset: g.offset.clone(),
            nbr: g.nbr.clone(),
            w: g.w.clone(),
            h: g.h.clone(),
            colors: g.colors.clone(),
            classes: g.classes.clone(),
            n_edges: g.n_edges,
        };
        let step = 1.0 / f64::from(1u32 << FRAC);
        for v in out.w.iter_mut().chain(out.h.iter_mut()) {
            let raw = *v;
            *v *= c;
            if raw != 0.0 && (*v).abs() < step / 2.0 {
                return Err(format!(
                    "a coefficient of {raw} scaled by {c} is {} and rounds to ZERO in Q.{FRAC}: the \
                     fabric would sample a problem with that coupling deleted",
                    *v
                ));
            }
        }
        Ok(out)
    }

    /// Put the registers at `beta`, charging the ledger only if they were somewhere else.
    fn set_temperature(&mut self, beta: f64) -> Result<(), String> {
        let g = self.graph.as_ref().ok_or("no program loaded")?;
        let c = beta / ROM_BETA;
        if self.held_scale == Some(c) {
            return Ok(());
        }
        let target = WritableRtl::scaled(g, c)?;
        let fab = self.fabric.as_mut().ok_or("no netlist synthesised")?;
        fab.program(&target).map_err(|e| format!("beta = {beta}: {e}"))?;
        self.held_scale = Some(c);
        self.ledger.writes += g.n as u64;
        Ok(())
    }

    /// Synthesise for `seed` if the netlist in hand carries another one.
    fn ensure_netlist(&mut self, seed: u64) -> Result<(), String> {
        let g = self.graph.as_ref().ok_or("no program loaded")?;
        if self.seed == Some(seed) {
            if let Some(f) = self.fabric.as_mut() {
                f.core.reset(); // the shell's soft reset: spins and generators, not weights
            }
            return Ok(());
        }
        // The power-on registers hold the program at ROM_BETA, which is what `program` loaded.
        let fab = WritableFabric::new(g, ROM_BETA, seed).map_err(|e| e.to_string())?;
        self.fabric = Some(fab);
        self.seed = Some(seed);
        self.held_scale = Some(1.0);
        if self.load_unused {
            self.load_unused = false;
        } else {
            self.ledger.writes += g.n as u64;
        }
        Ok(())
    }
}

impl Default for WritableRtl {
    fn default() -> Self {
        WritableRtl::new()
    }
}

impl crate::fabric::Device for WritableRtl {
    fn fabric(&self) -> crate::fabric::Fabric {
        let mut f = crate::fabric::Fabric::unconstrained(
            "ferrotherm-pbit-rtl-writable",
            crate::ledger::Prices::UNSTATED,
        );
        f.max_spins = self.max_spins;
        f.max_arity = 2;
        // No static coefficient range: a register holds beta*J, so whether a program fits depends
        // on the schedule it is run under, and `run` refuses the rung that does not.
        f
    }

    fn program(&mut self, p: &crate::ftp::Program) -> Vec<crate::fabric::Unsupported> {
        let mut bad = self.fabric().check(p);
        if !bad.is_empty() {
            return bad;
        }
        match p.to_graph() {
            Ok(g) => {
                if g.classes.len() != 2 {
                    bad.push(crate::fabric::Unsupported::Unplaceable {
                        detail: format!(
                            "this fabric updates two colour classes and the program needs {}",
                            g.classes.len()
                        ),
                    });
                    return bad;
                }
                self.state = vec![-1; g.n];
                self.ledger.writes += g.n as u64;
                self.load_unused = true;
                self.fabric = None;
                self.seed = None;
                self.held_scale = None;
                self.graph = Some(g);
            }
            Err(e) => bad.push(crate::fabric::Unsupported::Unplaceable { detail: e.to_string() }),
        }
        bad
    }

    fn run(&mut self, schedule: &crate::schedule::Schedule, seed: u64) -> Result<Vec<i8>, String> {
        if self.graph.is_none() {
            return Err("no program loaded".into());
        }
        if schedule.stages().is_empty() {
            return Err("a schedule with no stages advances nothing".into());
        }
        // Refuse the whole ladder BEFORE running any of it: a schedule whose sixth rung does not
        // fit must not leave five rungs of samples and writes on the ledger.
        {
            let g = self.graph.as_ref().expect("checked");
            for stage in schedule.stages() {
                let t = WritableRtl::scaled(g, stage.beta / ROM_BETA)?;
                WritableFabric::new(&t, ROM_BETA, 0).map_err(|e| format!("beta = {}: {e}", stage.beta))?;
            }
        }
        self.ensure_netlist(seed)?;
        let mut best: Option<Vec<i8>> = None;
        let mut best_e = f64::INFINITY;
        for stage in schedule.stages() {
            // A rung is a write to every node's registers -- and nothing else. No reset: the
            // spins and the generators are where the last rung left them.
            self.set_temperature(stage.beta)?;
            let g = self.graph.as_ref().expect("checked");
            let fab = self.fabric.as_mut().expect("synthesised");
            for _ in 0..stage.sweeps {
                fab.core.sweep();
                let st: Vec<i8> = fab.core.s.iter().map(|&up| if up { 1i8 } else { -1 }).collect();
                let e = g.energy(&st);
                if e < best_e {
                    best_e = e;
                    best = Some(st);
                }
            }
            self.ledger.samples += (g.n as u64) * (stage.sweeps as u64);
            self.ledger.reads += (g.n as u64) * (stage.sweeps as u64);
        }
        let fab = self.fabric.as_ref().expect("synthesised");
        self.state = best
            .unwrap_or_else(|| fab.core.s.iter().map(|&up| if up { 1i8 } else { -1 }).collect());
        Ok(self.state.clone())
    }

    fn sample(
        &mut self,
        beta: f64,
        plan: &crate::samples::Plan,
        seed: u64,
    ) -> Result<crate::samples::SampleSet, String> {
        if self.graph.is_none() {
            return Err("no program loaded".into());
        }
        self.ensure_netlist(seed)?;
        self.set_temperature(beta)?;
        let g = self.graph.as_ref().expect("checked");
        let fab = self.fabric.as_mut().expect("synthesised");
        for _ in 0..plan.burn_in {
            fab.core.sweep();
        }
        let thin = plan.thin.max(1);
        let mut states = Vec::with_capacity(plan.draws);
        let mut energies = Vec::with_capacity(plan.draws);
        for _ in 0..plan.draws {
            for _ in 0..thin {
                fab.core.sweep();
            }
            let st: Vec<i8> = fab.core.s.iter().map(|&up| if up { 1i8 } else { -1 }).collect();
            energies.push(g.energy(&st));
            states.push(st);
        }
        self.ledger.samples += (g.n as u64) * (plan.sweeps() as u64);
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
    use crate::ising::lattice2d;

    fn have_iverilog() -> bool {
        std::process::Command::new("iverilog").arg("-V").output().is_ok()
    }

    /// A second problem on the first problem's wiring: same lattice, weights that differ per edge
    /// and in sign, and a bias on every node — so a write that lands in the wrong slot, on the
    /// wrong node, or with the wrong sign produces a different trajectory.
    fn second_problem(side: usize) -> Graph {
        let mut g = lattice2d(side, 1.0);
        for i in 0..g.n {
            for k in g.offset[i]..g.offset[i + 1] {
                let j = g.nbr[k] as usize;
                let (a, b) = (i.min(j), i.max(j));
                // symmetric in (i, j) by construction, and different on every edge
                g.w[k] = (((a * 31 + b * 17) % 23) as f64 - 11.0) / 8.0;
            }
            g.h[i] = ((i * 7 % 13) as f64 - 6.0) / 10.0;
        }
        g
    }

    /// The emulator half of the contract, with no simulator needed: programming is the ONLY
    /// difference between a writable fabric and a fixed fabric built from the target.
    #[test]
    fn a_programmed_fabric_is_the_fixed_fabric_of_its_target() {
        let first = lattice2d(6, 0.9);
        let second = second_problem(6);
        let mut w = WritableFabric::new(&first, 0.4, 0xC0FFEE).expect("fits");
        let mut before = FixedFabric::new(&first, 0.4, 0xC0FFEE);
        let mut target = FixedFabric::new(&second, 0.4, 0xC0FFEE);
        for _ in 0..20 {
            w.core.sweep();
            before.sweep();
        }
        assert_eq!(w.core.s, before.s, "unprogrammed, it is the fabric it was synthesised from");

        let bus = w.program(&second).expect("same wiring, in range");
        assert_eq!(bus.len(), first.n * w.words_per_node());
        w.core.reset();
        let mut differed = false;
        for _ in 0..40 {
            w.core.sweep();
            target.sweep();
            before.sweep();
            assert_eq!(w.core.s, target.s, "programmed, it is the fixed fabric of the target");
            differed |= before.s != target.s;
        }
        assert!(differed, "vacuous if the two problems sample the same trajectory");
    }

    #[test]
    fn a_weight_that_does_not_fit_is_refused_and_nothing_is_written() {
        let g = lattice2d(4, 1.0);
        let mut w = WritableFabric::new(&g, 0.4, 1).expect("fits");
        let kept = w.core.adj.clone();
        let mut hot = lattice2d(4, 1.0);
        let last = hot.w.len() - 1;
        hot.w[last] = 8.0; // one step past +7.996
        assert!(matches!(w.program(&hot), Err(WriteError::Range { .. })));
        assert_eq!(w.core.adj, kept, "a refused program must not be half-written");
        hot.w[last] = 2047.0 / 256.0;
        assert!(w.program(&hot).is_ok(), "the largest representable weight is accepted");
        // Wires are not writable. A different node count is the easy case...
        let other = lattice2d(5, 1.0);
        assert!(matches!(w.program(&other), Err(WriteError::Topology { .. })));
        // ...and the one that matters is the SAME count wired differently: sixteen nodes in a ring
        // would be written into a torus's slots without complaint if only `n` were compared.
        let ring = crate::ising::ring(16, 1.0, 0.0);
        assert_eq!(ring.n, w.core.n);
        assert!(matches!(w.program(&ring), Err(WriteError::Topology { .. })));
        assert_eq!(w.core.adj[0].len(), 4, "and the refusal left the torus's wiring in place");
        assert!(WritableFabric::new(&hot, 0.4, 1).is_ok());
        hot.h[0] = -8.5;
        assert!(matches!(WritableFabric::new(&hot, 0.4, 1), Err(WriteError::Range { node: 0, .. })));
    }

    /// Temperature is a write: scaling every weight and bias by `c` IS the fixed fabric at `c·β`,
    /// wherever the scaled values are exact in Q.8.
    #[test]
    fn scaling_the_weights_is_a_change_of_temperature() {
        let g = lattice2d(6, 1.0);
        let mut doubled = lattice2d(6, 2.0);
        doubled.h.iter_mut().for_each(|h| *h *= 2.0);
        let mut w = WritableFabric::new(&g, 0.15, 9).expect("fits");
        w.program(&doubled).expect("fits");
        w.core.reset();
        let mut at_twice_beta = FixedFabric::new(&g, 0.30, 9);
        let mut unscaled = FixedFabric::new(&g, 0.15, 9);
        // Field 2f at beta and field f at 2*beta index the ROM at addresses whose midpoints differ
        // by a fraction of a Q.8 step, so the trajectories part and the STATISTICS are what agree.
        let sweeps = 20_000;
        let (mut ma, mut mb, mut mc) = (0.0, 0.0, 0.0);
        for _ in 0..sweeps {
            w.core.sweep();
            at_twice_beta.sweep();
            unscaled.sweep();
            ma += w.core.magnetization().abs();
            mb += at_twice_beta.magnetization().abs();
            mc += unscaled.magnetization().abs();
        }
        let (ma, mb, mc) = (ma / sweeps as f64, mb / sweeps as f64, mc / sweeps as f64);
        assert!((ma - mb).abs() < 0.01, "scaled weights |m| {ma} vs the fixed fabric at twice beta {mb}");
        // And the comparison could have failed: the unscaled fabric is somewhere else.
        assert!((mc - mb).abs() > 0.05, "vacuous: beta and 2*beta give |m| {mc} and {mb}");
    }

    /// A frustrated instance on the torus: every edge its own weight, both signs.
    fn glass(side: usize) -> Graph {
        let mut g = second_problem(side);
        g.h.iter_mut().for_each(|h| *h = 0.0);
        g
    }

    fn program_of(g: &Graph) -> crate::ftp::Program {
        crate::ftp::Program::from_graph(g, &crate::schedule::Schedule::constant(1.0, 1))
    }

    /// THE DEVICE'S CLAIM, in the two halves it actually has.
    #[test]
    fn a_ladder_on_the_writable_fabric_anneals_and_costs_one_more_write_not_fewer() {
        use crate::fabric::Device;
        use crate::schedule::Schedule;
        let g = glass(8);
        let n = g.n as u64;
        let p = program_of(&g);
        let ladder = Schedule::geometric(0.1, 3.0, 12, 8);
        let rungs = ladder.stages().len() as u64;

        // THE LEDGER. Same unit, and the writable fabric's count is HIGHER by one load.
        let mut fixed = crate::hdl::RtlFabric::new();
        let mut writable = WritableRtl::new();
        assert!(fixed.program(&p).is_empty() && writable.program(&p).is_empty());
        fixed.run(&ladder, 1).unwrap();
        writable.run(&ladder, 1).unwrap();
        assert_eq!(Device::ledger(&fixed).writes, n * rungs, "a bitstream a rung, the load being the first");
        assert_eq!(Device::ledger(&writable).writes, n * (rungs + 1), "a load, then a register write a rung");
        assert_eq!(Device::ledger(&fixed).samples, Device::ledger(&writable).samples);
        assert_eq!(Device::ledger(&fixed).reads, Device::ledger(&writable).reads);

        // CARRIED STATE, exactly: a rung boundary that changes nothing must be invisible. Two
        // rungs at one temperature are one rung of their total length, to the bit.
        let mut split = WritableRtl::new();
        let mut whole = WritableRtl::new();
        assert!(split.program(&p).is_empty() && whole.program(&p).is_empty());
        let mut two = Schedule::new();
        for sweeps in [9, 14] {
            two.push(crate::schedule::Stage { beta: 0.7, sweeps, penalties: Default::default() });
        }
        split.run(&two, 5).unwrap();
        whole.run(&Schedule::constant(0.7, 23), 5).unwrap();
        assert_eq!(
            split.fabric.as_ref().unwrap().core.s,
            whole.fabric.as_ref().unwrap().core.s,
            "a write-free rung boundary moved the chain"
        );
        assert_eq!(Device::ledger(&split).writes, Device::ledger(&whole).writes, "and an unchanged temperature is not charged twice");

        // WHAT CARRYING STATE BUYS. Twelve rungs of eight sweeps is ninety-six sweeps of annealing
        // here and twelve cold starts of eight sweeps there; same sweeps, same reads, same seeds.
        let (mut e_fixed, mut e_writable) = (0.0, 0.0);
        let seeds = 12;
        for seed in 0..seeds {
            let mut f = crate::hdl::RtlFabric::new();
            let mut w = WritableRtl::new();
            assert!(f.program(&p).is_empty() && w.program(&p).is_empty());
            e_fixed += g.energy(&f.run(&ladder, seed).unwrap());
            e_writable += g.energy(&w.run(&ladder, seed).unwrap());
        }
        let (e_fixed, e_writable) = (e_fixed / f64::from(seeds as u32), e_writable / f64::from(seeds as u32));
        eprintln!("mean best energy over {seeds} seeds: annealed {e_writable:.2}, twelve restarts {e_fixed:.2}");
        assert!(
            e_writable < e_fixed - 2.0,
            "annealing should beat restarts on a frustrated torus: {e_writable} vs {e_fixed}"
        );
    }

    #[test]
    fn a_rung_the_registers_cannot_hold_refuses_the_whole_ladder_before_running_any_of_it() {
        use crate::fabric::Device;
        use crate::schedule::Schedule;
        let g = glass(6);
        let p = program_of(&g);
        let mut d = WritableRtl::new();
        assert!(d.program(&p).is_empty());
        let after_load = Device::ledger(&d);
        // |w| reaches 1.375, so beta = 6 asks a register for 8.25.
        let too_cold = Schedule::geometric(0.2, 6.0, 6, 10);
        let err = d.run(&too_cold, 1).unwrap_err();
        assert!(err.contains("does not fit"), "{err}");
        let l = Device::ledger(&d);
        assert_eq!((l.samples, l.reads, l.writes), (after_load.samples, after_load.reads, after_load.writes), "five good rungs must not run before the sixth is refused");
        // And too HOT deletes couplings: 0.125 * 0.01 * 256 rounds to zero.
        let err = d.run(&Schedule::constant(0.01, 10), 1).unwrap_err();
        assert!(err.contains("rounds to ZERO"), "{err}");
        assert!(d.run(&Schedule::geometric(0.2, 5.0, 6, 10), 1).is_ok(), "and the ladder that fits, runs");
    }

    #[test]
    fn a_new_seed_is_a_new_netlist_and_is_charged_as_a_load() {
        use crate::fabric::Device;
        use crate::schedule::Schedule;
        let g = glass(6);
        let n = g.n as u64;
        let mut d = WritableRtl::new();
        assert!(d.program(&program_of(&g)).is_empty());
        let at_rom_beta = Schedule::constant(ROM_BETA, 5);
        let once = d.run(&at_rom_beta, 3).unwrap();
        let end_once = d.fabric.as_ref().unwrap().core.s.clone();
        assert_eq!(Device::ledger(&d).writes, n, "the load already holds the program at the ROM's beta");
        let again = d.run(&at_rom_beta, 3).unwrap();
        assert_eq!(Device::ledger(&d).writes, n, "the same seed is a soft reset, which writes no node");
        // A soft reset really is one: the second run is the first run, not its continuation.
        assert_eq!(once, again);
        assert_eq!(end_once, d.fabric.as_ref().unwrap().core.s, "a run must start from reset, not from where the last one stopped");
        d.run(&at_rom_beta, 4).unwrap();
        assert_eq!(Device::ledger(&d).writes, 2 * n, "seeds are reset constants of the netlist");
    }

    /// THE CARRY GATE. The device's ladder anneals only if the HARDWARE keeps its spins and its
    /// generators across a reprogram. So the emitted RTL is run to a target, rewritten over AXI
    /// with a second problem while stopped, and run on to a larger target WITH NO RESET — and must
    /// land on the emulator's state, which must in turn differ from what a reset would have given.
    #[test]
    fn rtl_reprogrammed_mid_run_without_a_reset_carries_its_state() {
        if !have_iverilog() {
            eprintln!("SKIP: iverilog not installed on this machine; the carry gate did not run, skipping");
            return;
        }
        // SHORT after the reprogram, and COLD. The first version ran eleven hot sweeps after it,
        // and a mutant RTL that RESET the chain on every configuration write still passed: the
        // run resumes at the same position in every generator's stream either way, and eleven
        // sweeps of shared random numbers at beta = 0.25 make two chains COALESCE whatever spins
        // they started from -- the mechanism `cftp` is built on, here erasing the very memory this
        // gate exists to see. It was testing that the generators carried, and nothing else.
        let (k, m) = (13usize, 2usize);
        let first = lattice2d(4, 0.9);
        let second = second_problem(4);
        let mut w = WritableFabric::new(&first, 0.6, 0xCA44).expect("fits");
        let core = w.emit_verilog("wfab");
        let shell = w.emit_axi_shell("wf_axi", "wfab", "clk");
        let word = |f: &FixedFabric| f.s.iter().enumerate().fold(0u32, |a, (i, &b)| a | (u32::from(b) << i));
        for _ in 0..k {
            w.core.sweep();
        }
        let bus = w.program(&second).expect("same wiring");
        for _ in 0..m {
            w.core.sweep();
        }
        let carried = word(&w.core);
        // THE TWO WAYS TO LOSE THE STATE, each of which must give a different answer or this
        // gate proves nothing. A reset that restarts the generators too (a reconfigured fixed
        // fabric)...
        let mut restarted = FixedFabric::new(&second, 0.6, 0xCA44);
        for _ in 0..m {
            restarted.sweep();
        }
        assert_ne!(carried, word(&restarted), "vacuous: a full restart reaches the same state");
        // ...and a reset after which the run RESUMES AT THE SAME STREAM POSITION, which is what a
        // shell that resets on a configuration write does, and what the first version could not
        // tell from carrying. Same random numbers, second problem throughout, spins from reset.
        let mut spins_lost = FixedFabric::new(&second, 0.6, 0xCA44);
        for _ in 0..k + m {
            spins_lost.sweep();
        }
        assert_ne!(carried, word(&spins_lost), "vacuous: the chains coalesced and the spins' memory is gone");
        let mut writes = String::new();
        for wr in &bus {
            writes.push_str(&format!("    wr(32'h{:08X}, 32'h{:08X});\n", wr.offset, wr.data));
        }
        let total = k + m;
        let tb = format!(
            r#"`timescale 1ns/1ps
module tb;
  reg clk = 0, rst_n = 0;
  reg [31:0] awaddr = 0, wdata = 0, araddr = 0;
  reg awvalid = 0, wvalid = 0, bready = 1, arvalid = 0, rready = 1;
  wire awready, wready, bvalid, arready, rvalid;
  wire [31:0] rdata; wire [1:0] bresp, rresp;
  wf_axi dut(.clk(clk), .rst_n(rst_n),
    .s_axi_awaddr(awaddr), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
    .s_axi_wdata(wdata), .s_axi_wstrb(4'hF), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
    .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
    .s_axi_araddr(araddr), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
    .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid), .s_axi_rready(rready));
  always #5 clk = ~clk;
  reg [31:0] got, got_done, got_status;
  integer guard;
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
  task wait_for(input [31:0] want);
    begin
      guard = 0; got_status = 0;
      while ((got_status[1] !== 1'b1) && guard < 100000) begin rd(32'h04, got_status); guard = guard + 1; end
      rd(32'h0C, got_done);
      if (got_done !== want) begin $display("FERROTHERM_FAIL sweeps %0d want %0d", got_done, want); $finish; end
    end
  endtask
  initial begin #20000000; $display("FERROTHERM_FAIL watchdog"); $finish; end
  initial begin
    repeat (4) @(posedge clk); rst_n = 1; repeat (2) @(posedge clk);
    wr(32'h08, 32'd{k}); wr(32'h00, 32'h1); wait_for(32'd{k});
    // stopped at the target: rewrite every node, and do NOT reset
{writes}    wr(32'h08, 32'd{total}); wait_for(32'd{total});
    rd(32'h20, got);
    if (got !== 32'h{carried:08x}) $display("FERROTHERM_FAIL state %h want {carried:08x}", got);
    else $display("FERROTHERM_PASS");
    $finish;
  end
endmodule
"#
        );
        let dir = std::env::temp_dir().join(format!("ferrotherm_carry_{}", std::process::id()));
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
        assert!(stdout.contains("FERROTHERM_PASS"), "carry gate:\n{stdout}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A bus master that behaves BADLY, within the rules: address before data, data before
    /// address, both at once; a reader running concurrently with the writer; `bready` and `rready`
    /// held low for random stretches. Every transaction must complete inside a bound, every read
    /// must return what was last written, and the shell's count of configuration words must equal
    /// the count sent.
    ///
    /// Written on 2026-09-19 after a KV260 stopped answering minutes after this shell's first run
    /// on it. The polite testbench above issues one transaction at a time with `bready` tied high,
    /// which is not how an interconnect behaves; a slave that deadlocks under overlap stalls the
    /// core that issued the transaction, for ever, and takes the machine with it.
    fn stress(shell: &str, top: &str, core: &str, has_config: bool) -> String {
        let cfg = if has_config { 1 } else { 0 };
        let tb = format!(
            r#"`timescale 1ns/1ps
module tb;
  reg clk = 0, rst_n = 0;
  reg [31:0] awaddr = 0, wdata = 0, araddr = 0;
  reg awvalid = 0, wvalid = 0, bready = 0, arvalid = 0, rready = 0;
  wire awready, wready, bvalid, arready, rvalid;
  wire [31:0] rdata; wire [1:0] bresp, rresp;
  {top} dut(.clk(clk), .rst_n(rst_n),
    .s_axi_awaddr(awaddr), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
    .s_axi_wdata(wdata), .s_axi_wstrb(4'hF), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
    .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
    .s_axi_araddr(araddr), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
    .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid), .s_axi_rready(rready));
  always #5 clk = ~clk;
  integer seed = 32'h5EED1234, k, d_aw, d_w, d_b, cfg_sent = 0, reads_done = 0, writes_done = 0;
  reg [31:0] last_target = 0, got, next_addr, next_data;
  reg writer_done = 0;
  localparam LIMIT = 400;
  task fail(input [255:0] why); begin $display("FERROTHERM_FAIL %0s (after %0d writes, %0d reads)", why, writes_done, reads_done); $finish; end endtask
  // One driver per channel, each obeying the one rule a master has: hold VALID until READY.
  task drive_aw(input [31:0] a, input integer delay); integer n; begin
    repeat (delay) @(posedge clk);
    awaddr <= a; awvalid <= 1; n = 0;
    @(posedge clk); while (!awready) begin @(posedge clk); n = n + 1; if (n > LIMIT) fail("a write address was never accepted"); end
    awvalid <= 0;
  end endtask
  task drive_w(input [31:0] x, input integer delay); integer n; begin
    repeat (delay) @(posedge clk);
    wdata <= x; wvalid <= 1; n = 0;
    @(posedge clk); while (!wready) begin @(posedge clk); n = n + 1; if (n > LIMIT) fail("write data was never accepted"); end
    wvalid <= 0;
  end endtask
  // THE WRITER: address first, data first, or together, by up to seven clocks either way; and a
  // response accepted late.
  initial begin
    repeat (4) @(posedge clk); rst_n = 1; repeat (2) @(posedge clk);
    for (k = 0; k < 3000; k = k + 1) begin
      d_aw = $urandom(seed) % 8; d_w = $urandom(seed) % 8; d_b = $urandom(seed) % 6;
      if ({cfg} && ($urandom(seed) % 2)) begin next_addr = 32'h1000 + 4 * ($urandom(seed) % 8); next_data = $urandom(seed); cfg_sent = cfg_sent + 1; end
      else begin next_addr = 32'h08; next_data = 32'h1000 + k; last_target = next_data; end
      fork
        drive_aw(next_addr, d_aw);
        drive_w(next_data, d_w);
      join
      repeat (d_b) @(posedge clk);
      bready <= 1; d_b = 0;
      @(posedge clk); while (!bvalid) begin @(posedge clk); d_b = d_b + 1; if (d_b > LIMIT) fail("a write response never came"); end
      bready <= 0;
      writes_done = writes_done + 1;
    end
    writer_done = 1;
  end
  // THE READER, concurrently: the target register must read back a value the writer has written.
  integer rw, rd_gap;
  initial begin
    repeat (10) @(posedge clk);
    while (!writer_done) begin
      rd_gap = $urandom(seed) % 5; repeat (rd_gap) @(posedge clk);
      araddr <= ($urandom(seed) % 2) ? 32'h08 : 32'h04; arvalid <= 1; rw = 0;
      @(posedge clk); while (!arready) begin @(posedge clk); rw = rw + 1; if (rw > LIMIT) fail("a read address was never accepted"); end
      arvalid <= 0;
      rd_gap = $urandom(seed) % 6; repeat (rd_gap) @(posedge clk);
      rready <= 1; rw = 0;
      @(posedge clk); while (!rvalid) begin @(posedge clk); rw = rw + 1; if (rw > LIMIT) fail("read data never came"); end
      rready <= 0; reads_done = reads_done + 1;
    end
    // quiesce, then the two facts with known answers
    repeat (20) @(posedge clk);
    araddr <= 32'h08; arvalid <= 1; @(posedge clk); while (!arready) @(posedge clk); arvalid <= 0;
    rready <= 1; @(posedge clk); while (!rvalid) @(posedge clk); got = rdata; rready <= 0;
    if (got !== last_target) fail("the target register does not hold the last value written");
    if ({cfg}) begin
      @(posedge clk); araddr <= 32'h14; arvalid <= 1; @(posedge clk); while (!arready) @(posedge clk); arvalid <= 0;
      rready <= 1; @(posedge clk); while (!rvalid) @(posedge clk); got = rdata; rready <= 0;
      if (got !== cfg_sent) fail("configuration words accepted differ from words sent");
    end
    $display("FERROTHERM_PASS %0d writes %0d reads %0d config", writes_done, reads_done, cfg_sent);
    $finish;
  end
  initial begin #60000000; $display("FERROTHERM_FAIL watchdog: %0d writes %0d reads", writes_done, reads_done); $finish; end
endmodule
"#
        );
        let dir = std::env::temp_dir().join(format!("ferrotherm_stress_{top}_{}", std::process::id()));
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
        let _ = std::fs::remove_dir_all(&dir);
        String::from_utf8_lossy(&run.stdout).into_owned()
    }

    #[test]
    fn neither_shell_deadlocks_under_a_badly_behaved_master() {
        if !have_iverilog() {
            eprintln!("SKIP: iverilog not installed on this machine; the bus stress gate did not run, skipping");
            return;
        }
        let g = lattice2d(4, 0.9);
        let w = WritableFabric::new(&g, 0.25, 7).expect("fits");
        let out = stress(&w.emit_axi_shell("wf_axi", "wfab", "clk"), "wf_axi", &w.emit_verilog("wfab"), true);
        assert!(out.contains("FERROTHERM_PASS"), "writable shell under stress:\n{out}");
        let f = FixedFabric::new(&g, 0.25, 7);
        let out = stress(&f.emit_axi_shell("ft_axi", "fabric", "clk"), "ft_axi", &f.emit_verilog("fabric"), false);
        assert!(out.contains("FERROTHERM_PASS"), "fixed shell under stress:\n{out}");
    }

    /// THE HARDWARE GATE. The emitted RTL is driven over its AXI port as a host would drive it:
    /// run the power-on problem and read the state; write a SECOND problem word by word; reset
    /// the chain; run again and read again. Both reads must equal a `FixedFabric` built directly
    /// from each problem — and the two must differ, or the writes proved nothing.
    #[test]
    fn rtl_written_over_axi_matches_the_fixed_fabric_of_what_was_written() {
        if !have_iverilog() {
            eprintln!("SKIP: iverilog not installed on this machine; the writable-fabric gate did not run, skipping");
            return;
        }
        let sweeps = 20usize;
        let first = lattice2d(4, 0.9);
        let second = second_problem(4);
        let mut w = WritableFabric::new(&first, 0.25, 0x5EED).expect("fits");
        let core = w.emit_verilog("wfab");
        let shell = w.emit_axi_shell("wf_axi", "wfab", "clk");
        let word = |f: &FixedFabric| f.s.iter().enumerate().fold(0u32, |a, (i, &b)| a | (u32::from(b) << i));
        let mut a = FixedFabric::new(&first, 0.25, 0x5EED);
        let mut b = FixedFabric::new(&second, 0.25, 0x5EED);
        for _ in 0..sweeps {
            a.sweep();
            b.sweep();
        }
        let (want_a, want_b) = (word(&a), word(&b));
        assert_ne!(want_a, want_b, "vacuous unless the second problem reaches a different state");
        let bus = w.program(&second).expect("same wiring");
        let mut writes = String::new();
        for wr in &bus {
            writes.push_str(&format!("    wr(32'h{:08X}, 32'h{:08X});\n", wr.offset, wr.data));
        }
        let n_writes = bus.len();

        let tb = format!(
            r#"`timescale 1ns/1ps
module tb;
  reg clk = 0, rst_n = 0;
  reg [31:0] awaddr = 0, wdata = 0, araddr = 0;
  reg awvalid = 0, wvalid = 0, bready = 1, arvalid = 0, rready = 1;
  wire awready, wready, bvalid, arready, rvalid;
  wire [31:0] rdata; wire [1:0] bresp, rresp;
  wf_axi dut(.clk(clk), .rst_n(rst_n),
    .s_axi_awaddr(awaddr), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
    .s_axi_wdata(wdata), .s_axi_wstrb(4'hF), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
    .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
    .s_axi_araddr(araddr), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
    .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid), .s_axi_rready(rready));
  always #5 clk = ~clk;
  reg [31:0] got_a, got_b, got_done, got_status, got_cfg;
  integer guard;
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
  task run_to_target;
    begin
      wr(32'h00, 32'h2);            // reset the chain; the weights stay
      wr(32'h08, 32'd{sweeps});
      wr(32'h00, 32'h1);
      guard = 0; got_status = 0;
      while ((got_status[1] !== 1'b1) && guard < 100000) begin
        rd(32'h04, got_status); guard = guard + 1;
      end
      if (got_status[1] !== 1'b1) begin $display("FERROTHERM_FAIL never reached target"); $finish; end
      rd(32'h0C, got_done);
      if (got_done !== 32'd{sweeps}) begin $display("FERROTHERM_FAIL sweeps %0d want {sweeps}", got_done); $finish; end
      wr(32'h00, 32'h0);
    end
  endtask
  initial begin
    #20000000;
    $display("FERROTHERM_FAIL watchdog");
    $finish;
  end
  initial begin
    repeat (4) @(posedge clk); rst_n = 1; repeat (2) @(posedge clk);
    run_to_target; rd(32'h20, got_a);
{writes}    rd(32'h14, got_cfg);
    run_to_target; rd(32'h20, got_b);
    if (got_a !== 32'h{want_a:08x})      $display("FERROTHERM_FAIL power-on problem: state %h want {want_a:08x}", got_a);
    else if (got_cfg !== 32'd{n_writes}) $display("FERROTHERM_FAIL %0d configuration words accepted, want {n_writes}", got_cfg);
    else if (got_b !== 32'h{want_b:08x}) $display("FERROTHERM_FAIL written problem: state %h want {want_b:08x}", got_b);
    else $display("FERROTHERM_PASS");
    $finish;
  end
endmodule
"#
        );
        let dir = std::env::temp_dir().join(format!("ferrotherm_wfab_{}", std::process::id()));
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
        assert!(stdout.contains("FERROTHERM_PASS"), "writable fabric gate:\n{stdout}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
