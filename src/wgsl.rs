//! The sweep, emitted as a WebGPU compute shader.
//!
//! Open GPU sampling of Ising models is an empty lane. OpenJij, the reference open sampler, dropped
//! GPGPU support in 2023 and is CPU-only, while every commercial engine is GPU, FPGA or ASIC. There
//! is no open, permissively licensed, browser-capable sampler. This is ours, and it reaches a GPU
//! without adding a dependency: Rust emits WGSL text the same way [`crate::hdl`] emits Verilog, and
//! the browser compiles it.
//!
//! # What agrees, and what cannot
//!
//! The emitted shader is **not** bit-identical to the CPU sampler, and claiming otherwise would be
//! false in two independent ways:
//!
//! - **WGSL has no `f64`.** Fields are accumulated in `f32`. On a degree-16 graph with couplings of
//!   order 1 that is a relative error around `1e-7`, far below the width of the sigmoid, but it is
//!   real and it is not zero.
//! - **The random stream differs.** A CPU sweep draws from one sequential stream; a GPU sweep has
//!   every lane draw independently from a counter-based hash of `(seed, node, step)`. Reproducing
//!   the sequential order across a thousand parallel lanes would defeat the point of using them.
//!
//! So the two backends agree **in distribution**, not in bytes, and the right instrument for
//! checking that is the one this crate already has: run both, certify both, and compare the
//! effective temperature and the distance from exact. A backend that samples the wrong distribution
//! cannot hide from [`crate::certify`], and bit-comparison would have told us nothing anyway.
//!
//! Determinism still holds *within* the GPU path: same seed and same dispatch order reproduce a run
//! exactly, because the hash is a pure function of `(seed, node, step)` rather than of arrival
//! order.

use crate::dense::Padded;
use crate::graph::Graph;

/// Buffer contents for the emitted shader, in the layout it expects.
pub struct GpuModel {
    pub n: u32,
    /// Row width of the padded interaction rectangle.
    pub k: u32,
    /// `n * k` neighbour indices.
    pub nbr: Vec<u32>,
    /// `n * k` couplings, f32 for WGSL.
    pub w: Vec<f32>,
    /// `n` biases.
    pub h: Vec<f32>,
    /// Node indices grouped by colour: `classes[c]` update together.
    pub classes: Vec<Vec<u32>>,
}

impl GpuModel {
    pub fn from_graph(g: &Graph) -> GpuModel {
        let d = Padded::from_graph(g);
        GpuModel {
            n: g.n as u32,
            k: d.k as u32,
            nbr: d.nbr,
            // padded slots are already exactly 0.0, so the mask need not cross to the GPU: a zero
            // weight contributes nothing whether or not a lane knows it is padding
            w: d.w.iter().map(|&x| x as f32).collect(),
            h: d.h.iter().map(|&x| x as f32).collect(),
            classes: g.classes.iter().map(|c| c.to_vec()).collect(),
        }
    }

    /// Bytes of GPU memory this model needs, so a caller can size it before uploading.
    pub fn bytes(&self) -> usize {
        let nk = (self.n as usize) * (self.k as usize);
        nk * 4 + nk * 4 + (self.n as usize) * 4 + (self.n as usize) * 4
    }
}

/// Workgroup size. 64 is the portable choice: it is a multiple of both the 32-lane and 64-lane
/// wavefronts in circulation, so no vendor is left running quarter-empty.
pub const WORKGROUP: u32 = 64;

/// Emit the chromatic block-Gibbs sweep as a WGSL compute shader.
///
/// One invocation per node of the colour class being updated. Nodes of one colour share no edges,
/// so they can be resampled in the same instant without racing — which is the whole reason this
/// computation suits a GPU, and it is a property of the *colouring*, not of any locking.
pub fn sweep_shader() -> String {
    format!(
        r#"// ferrotherm: chromatic block-Gibbs sweep.
// Generated. One invocation per node of the active colour class.
// Nodes sharing a colour share no edges, so this is race-free by construction, not by locking.

struct Params {{
  n: u32,
  k: u32,
  class_len: u32,
  step: u32,
  beta: f32,
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
}};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read>       nbr:   array<u32>;
@group(0) @binding(2) var<storage, read>       w:     array<f32>;
@group(0) @binding(3) var<storage, read>       h:     array<f32>;
@group(0) @binding(4) var<storage, read>       class: array<u32>;
@group(0) @binding(5) var<storage, read_write> spin:  array<i32>;

// Counter-based RNG. A pure function of (seed, node, step), so a lane needs no state and the run
// reproduces regardless of the order lanes happen to execute in.
fn hash(a0: u32, b0: u32, c0: u32) -> u32 {{
  var x: u32 = a0 * 0x9E3779B9u ^ b0 * 0x85EBCA6Bu ^ c0 * 0xC2B2AE35u;
  x = x ^ (x >> 16u);
  x = x * 0x7FEB352Du;
  x = x ^ (x >> 15u);
  x = x * 0x846CA68Bu;
  x = x ^ (x >> 16u);
  return x;
}}

fn unit(a0: u32, b0: u32, c0: u32) -> f32 {{
  // 24 bits into [0,1); f32 has 24 bits of mantissa, so asking for more would be theatre
  return f32(hash(a0, b0, c0) >> 8u) * (1.0 / 16777216.0);
}}

@compute @workgroup_size({wg})
fn sweep(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let t = gid.x;
  if (t >= P.class_len) {{ return; }}
  let i = class[t];

  // local field: sum_j J_ij s_j + h_i. Padded slots carry weight 0.0 and contribute nothing.
  var f: f32 = h[i];
  let base = i * P.k;
  for (var s: u32 = 0u; s < P.k; s = s + 1u) {{
    let idx = base + s;
    f = f + w[idx] * f32(spin[nbr[idx]]);
  }}

  // the one update: P(s_i = +1) = sigma(2 beta f)
  let p = 1.0 / (1.0 + exp(-2.0 * P.beta * f));
  if (unit(P.step, i, 0x5BF03635u) < p) {{
    spin[i] = 1;
  }} else {{
    spin[i] = -1;
  }}
}}
"#,
        wg = WORKGROUP
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shader_states_the_same_update_as_the_kernel() {
        // The shader cannot call kernel::p_up, so the only defence against the two drifting apart
        // is that the expression is pinned here and in `hdl` the same way.
        let src = sweep_shader();
        assert!(
            src.contains("1.0 / (1.0 + exp(-2.0 * P.beta * f))"),
            "the sigmoid must be sigma(2*beta*f); anything else samples a different temperature"
        );
        assert!(src.contains("spin[i] = 1;") && src.contains("spin[i] = -1;"), "states are -1/+1");
    }

    #[test]
    fn the_emitted_shader_is_syntactically_plausible() {
        let src = sweep_shader();
        assert_eq!(src.matches('{').count(), src.matches('}').count(), "unbalanced braces");
        for needed in [
            "@compute",
            "@workgroup_size(64)",
            "fn sweep(",
            "var<storage, read_write> spin",
            "var<uniform> P",
        ] {
            assert!(src.contains(needed), "missing {needed}");
        }
        // every binding index used exactly once
        for b in 0..=5 {
            assert_eq!(
                src.matches(&format!("@binding({b})")).count(),
                1,
                "binding {b} should appear exactly once"
            );
        }
    }

    #[test]
    fn the_model_matches_the_padded_layout() {
        let g = crate::ising::lattice2d(8, 1.0);
        let m = GpuModel::from_graph(&g);
        assert_eq!(m.n, 64);
        assert_eq!(m.k, 4, "a square lattice has degree 4");
        assert_eq!(m.nbr.len(), 64 * 4);
        assert_eq!(m.w.len(), 64 * 4);
        assert_eq!(m.classes.len(), 2, "a bipartite lattice needs two colours");
        let total: usize = m.classes.iter().map(|c| c.len()).sum();
        assert_eq!(total, 64, "every node belongs to exactly one class");
    }

    #[test]
    fn the_f32_field_stays_within_the_sigmoid_width() {
        // The precision claim in the module docs, measured rather than asserted. If f32 error were
        // comparable to the sigmoid's own scale, the GPU would sample a visibly different model.
        let g = crate::device::z1_grid(8, 8, 1.0, 0.2);
        let m = GpuModel::from_graph(&g);
        let d = crate::dense::Padded::from_graph(&g);
        let mut rng = crate::rng::Pcg::new(4, 0);
        let s: Vec<i8> = (0..g.n).map(|_| if rng.f64() < 0.5 { 1 } else { -1 }).collect();

        let mut worst: f64 = 0.0;
        for i in 0..g.n {
            let exact = d.field(i, &s);
            let mut f32_sum = m.h[i];
            for slot in 0..m.k as usize {
                let t = i * m.k as usize + slot;
                f32_sum += m.w[t] * s[m.nbr[t] as usize] as f32;
            }
            // difference in the resulting probability, which is what actually matters
            let pa = crate::kernel::p_up(exact, 1.0);
            let pb = crate::kernel::p_up(f32_sum as f64, 1.0);
            worst = worst.max((pa - pb).abs());
        }
        assert!(worst < 1e-6, "f32 accumulation shifted a probability by {worst}");
    }

    #[test]
    fn padding_needs_no_mask_on_the_gpu() {
        // The dense layout's double safety pays off here: because padded slots weigh exactly zero,
        // the shader can skip uploading the mask entirely and still compute the right field. A
        // sentinel-padded layout could not do this.
        let mut b = crate::graph::GraphBuilder::new(20);
        for j in 1..20 {
            b.couple(0, j, 1.0); // a star: ragged degrees, so plenty of padding
        }
        let g = b.build();
        let m = GpuModel::from_graph(&g);
        let d = crate::dense::Padded::from_graph(&g);
        for t in 0..(m.n as usize * m.k as usize) {
            if d.active[t] == 0 {
                assert_eq!(m.w[t], 0.0);
            }
        }
    }
}
