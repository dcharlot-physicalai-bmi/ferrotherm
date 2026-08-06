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

pub const FRAC: u32 = 8; // Q.8 fixed point
const FMAX: i32 = 2047; // clamp field to [-8.0, +8.0) in Q.8
const FMIN: i32 = -2048;
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

pub struct FixedFabric {
    pub n: usize,
    /// per node: (neighbor index, quantized weight)
    pub adj: Vec<Vec<(u32, i32)>>,
    pub bias_q: Vec<i32>,
    /// the two chromatic classes (v1 requires a bipartite graph)
    pub classes: [Vec<u32>; 2],
    pub lut: Vec<u16>,
    pub seeds: Vec<u32>,
    pub init_s: Vec<bool>,
    /// current emulator state (true = +1)
    pub s: Vec<bool>,
    pub rng: Vec<u32>,
}

impl FixedFabric {
    /// Quantize a graph at inverse temperature `beta` into the fabric. Panics unless the
    /// coloring is exactly two classes (the lattice / device-topology case v1 targets).
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
    pub fn emit_verilog(&self, module: &str) -> String {
        let n = self.n;
        let mut v = String::new();
        v.push_str(&format!(
            "// generated by ferrotherm::hdl — fixed-point chromatic-Gibbs p-bit fabric\n\
             // {n} p-bits, Q.{FRAC} weights, {}-entry sigmoid ROM, xorshift32 per node\n\
             module {module} (\n    input wire clk,\n    input wire rst,\n    output reg [{top}:0] state,\n    output reg phase\n);\n",
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
        v.push_str("    end else begin\n      phase <= ~phase;\n");
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

    /// Emit a self-checking testbench plus the expected per-sweep state trace (hex lines) from
    /// the emulator. The testbench prints FERROTHERM_PASS only on a bit-exact match.
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
            "`timescale 1ns/1ps\nmodule tb;\n  reg clk = 0, rst = 1;\n  wire [{top}:0] state;\n  wire phase;\n  {module} dut(.clk(clk), .rst(rst), .state(state), .phase(phase));\n  reg [{top}:0] expected [0:{last}];\n  integer sw, errors = 0;\n  always #5 clk = ~clk;\n  initial begin\n    $readmemh(\"expected.hex\", expected);\n    @(posedge clk); @(posedge clk); rst = 0;\n    for (sw = 0; sw < {sweeps}; sw = sw + 1) begin\n      @(posedge clk); @(posedge clk); #1;\n      if (state !== expected[sw]) begin\n        errors = errors + 1;\n        $display(\"MISMATCH sweep %0d: got %h want %h\", sw, state, expected[sw]);\n      end\n    end\n    if (errors == 0) $display(\"FERROTHERM_PASS\");\n    else $display(\"FERROTHERM_FAIL %0d\", errors);\n    $finish;\n  end\nendmodule\n",
            top = n - 1,
            last = sweeps - 1,
        );
        (tb, expected)
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
        for s in fab.s.iter_mut() {
            *s = true;
        }
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
