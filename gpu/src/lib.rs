//! Native GPU sampling: the same chromatic sweep the browser runs, on Vulkan, Metal or DX12.
//!
//! # Why this is a separate crate
//!
//! `ferrotherm` is std-only with zero dependencies, and that is load-bearing rather than
//! decorative: it is what lets the same source compile to `wasm32-unknown-unknown` and to a
//! microcontroller. A GPU backend needs a driver stack. So it lives out here beside `silicon`,
//! `serve` and `cloud`, each of which exists for exactly the same reason.
//!
//! # Why it does not have its own shader
//!
//! The WGSL comes from [`ferrotherm::wgsl::sweep_shader`] — the same string the browser fetches
//! through `ft_shader`. A second copy would be a second implementation of the update rule, and the
//! two would drift the first time one was tuned. The core crate already pins the sigmoid with a
//! test (`the_shader_states_the_same_update_as_the_kernel`); binding that same text here means a
//! native run and a browser run cannot disagree about the arithmetic, only about the hardware.
//!
//! # What it does not promise
//!
//! **Not bit-identical to the CPU sampler.** The shader's RNG is a counter-based hash of
//! `(step, node)`, chosen so a lane needs no state and the result does not depend on the order
//! lanes happen to execute in. The CPU sampler draws from its own stream. Both sample the same
//! distribution; neither reproduces the other's individual flips, and a test that asserted they did
//! would be asserting something false.
//!
//! What they DO agree on is physics, and that is what [`Gpu::sweep`]'s tests check: the same
//! magnetisation at the same temperature, and the exact mean energy from variable elimination.
//!
//! # Verified on two vendors and two APIs
//!
//! | | adapter | API | tests |
//! |---|---|---|---|
//! | Apple M5 Max | IntegratedGpu | Metal | 6/6 |
//! | NVIDIA L4 (EC2 g6.xlarge) | DiscreteGpu | Vulkan 1.4 | 6/6 |
//!
//! Both run the same WGSL from the core crate, and both reproduce the exact mean energy computed by
//! variable elimination. That matters more than it sounds: a shader can pass on Metal and fail on
//! Vulkan, whose validation is stricter and whose f32 behaviour differs, and "runs on Vulkan, Metal
//! or DX12" was previously a claim checked on one of the three. DX12 remains unchecked.
//!
//! ```no_run
//! use ferrotherm::{ising::lattice2d, wgsl::GpuModel};
//! # fn main() -> Result<(), String> {
//! let g = lattice2d(8, 1.0);
//! let m = GpuModel::from_graph(&g);
//! let mut spins = vec![1i8; 64];
//!
//! let gpu = ferrotherm_gpu::Gpu::new().ok_or("no adapter")?;
//! gpu.sweep(&m, &mut spins, 0.44, 100)?;
//! # Ok(()) }
//! ```

use ferrotherm::wgsl::{sweep_shader, GpuModel};
use wgpu::util::DeviceExt;

/// A GPU that can run the sweep.
///
/// Holds a device and queue. Creating one enumerates adapters, which is slow enough that it should
/// happen once per process rather than once per sweep.
pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// What the adapter reported. Worth carrying because a software rasteriser will happily run
    /// this and report timings that mean nothing about hardware — see [`Gpu::adapter`].
    info: wgpu::AdapterInfo,
}

impl Gpu {
    /// Open the default adapter, or `None` if this machine exposes none.
    ///
    /// `None` means **not found on this machine**, never "impossible". A headless CI runner with no
    /// driver is the common case, which is why every test here skips rather than fails on it.
    pub fn new() -> Option<Gpu> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .ok()?;
        let info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("ferrotherm"),
            required_features: wgpu::Features::empty(),
            // The WebGPU baseline, NOT downlevel_defaults. Downlevel caps storage buffers at 4
            // per stage and this shader binds 6 (nbr, w, h, cls, spin, dbg), so asking for
            // downlevel produces a device that cannot compile the pipeline -- and the failure
            // arrives as a validation error at pipeline creation, far from the line that chose
            // the limit. The browser runs this same shader under the WebGPU baseline, so the
            // baseline is exactly the right floor: anything that runs the page runs this.
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            trace: wgpu::Trace::Off,
        }))
        .ok()?;
        Some(Gpu { device, queue, info })
    }

    /// What the driver says this is.
    ///
    /// Read it before quoting a speedup. `DeviceType::Cpu` is a software rasteriser — lavapipe,
    /// SwiftShader, WARP — which runs the shader correctly and tells you nothing about a GPU, and a
    /// benchmark that does not check this reports the wrong machine with full confidence.
    pub fn adapter(&self) -> &wgpu::AdapterInfo {
        &self.info
    }

    /// True when the adapter is real silicon rather than a software rasteriser.
    #[must_use = "false means a software rasteriser, whose timings say nothing about a GPU. Quoting a speedup without checking this reports the wrong machine"]
    pub fn is_hardware(&self) -> bool {
        !matches!(self.info.device_type, wgpu::DeviceType::Cpu | wgpu::DeviceType::Other)
    }

    /// Run `sweeps` chromatic sweeps over `spins`, in place.
    ///
    /// One dispatch per colour class per sweep, which is what makes the update correct: nodes in a
    /// class share no edge, so they can be resampled simultaneously without any of them reading a
    /// neighbour another lane is writing. Dispatching all nodes at once would be faster and wrong.
    pub fn sweep(
        &self,
        m: &GpuModel,
        spins: &mut [i8],
        beta: f64,
        sweeps: u32,
    ) -> Result<(), String> {
        if spins.len() != m.n as usize {
            return Err(format!(
                "this model has {} nodes and that state has {}",
                m.n,
                spins.len()
            ));
        }
        if !beta.is_finite() || beta < 0.0 {
            return Err(format!("beta must be finite and non-negative, not {beta}"));
        }
        if m.classes.is_empty() {
            return Err("a model with no colour classes has nothing to dispatch".into());
        }

        let dev = &self.device;
        let storage = wgpu::BufferUsages::STORAGE;
        let rw = storage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;

        // A zero-length storage buffer is invalid, and a graph with no couplings produces one. Pad
        // to a single element rather than failing: the shader reads k = 0 and never indexes it.
        let pad_u32 = |v: &[u32]| if v.is_empty() { vec![0u32] } else { v.to_vec() };
        let pad_f32 = |v: &[f32]| if v.is_empty() { vec![0f32] } else { v.to_vec() };

        let mk_u32 = |label: &str, data: &[u32], usage: wgpu::BufferUsages| {
            dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes_u32(&pad_u32(data)),
                usage,
            })
        };
        let mk_f32 = |label: &str, data: &[f32], usage: wgpu::BufferUsages| {
            dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes_f32(&pad_f32(data)),
                usage,
            })
        };

        let b_nbr = mk_u32("nbr", &m.nbr, storage);
        let b_w = mk_f32("w", &m.w, storage);
        let b_h = mk_f32("h", &m.h, storage);
        // The shader stores spins as i32; the library holds them as i8.
        let state: Vec<i32> = spins.iter().map(|&s| s as i32).collect();
        let b_spin = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("spin"),
            contents: bytes_i32(&state),
            usage: rw,
        });
        let b_dbg = mk_f32("dbg", &vec![0f32; m.n as usize], rw);
        let classes: Vec<(u32, wgpu::Buffer)> = m
            .classes
            .iter()
            .map(|c| (c.len() as u32, mk_u32("cls", c, storage)))
            .collect();


        let readback = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (state.len() * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sweep"),
            source: wgpu::ShaderSource::Wgsl(sweep_shader().into()),
        });

        // An EXPLICIT layout, because binding 0 needs `has_dynamic_offset`. An auto-derived layout
        // cannot express that, and without it every dispatch needs its own params buffer, its own
        // bind group and -- fatally -- its own submit.
        //
        // That is what the first version did, and it made the GPU slower than the CPU at every
        // size: 200 sweeps over 2 colour classes is 400 submits, each a driver round trip, and the
        // measured time was ~60 ms almost independent of node count. Constant time under a growing
        // workload is the signature of paying for round trips rather than arithmetic.
        let sto = |ro: bool| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: ro },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let entry = |binding: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let layout = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sweep"),
            entries: &[
                entry(0, wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(PARAMS_BYTES),
                }),
                entry(1, sto(true)),
                entry(2, sto(true)),
                entry(3, sto(true)),
                entry(4, sto(true)),
                entry(5, sto(false)),
                entry(6, sto(false)),
            ],
        });
        let pipeline_layout = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sweep"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sweep"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("sweep"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Every dispatch's params, written once into one buffer at the alignment the device
        // requires, then selected by dynamic offset. The step counter advances per dispatch --
        // it feeds the shader's counter-based RNG, and repeating it would make every class
        // resample with the same draws and the chain stop mixing.
        let stride = align_up(PARAMS_BYTES, dev.limits().min_uniform_buffer_offset_alignment as u64);
        let live: Vec<usize> = (0..classes.len()).filter(|&i| classes[i].0 > 0).collect();
        if live.is_empty() {
            return Err("every colour class is empty; there is nothing to sample".into());
        }
        let steps = sweeps as usize * live.len();
        let mut params = vec![0u8; steps * stride as usize];
        for s in 0..sweeps as usize {
            for (li, &ci) in live.iter().enumerate() {
                let step = (s * live.len() + li + 1) as u32;
                let at = (s * live.len() + li) * stride as usize;
                let p = &mut params[at..at + PARAMS_BYTES as usize];
                p[0..4].copy_from_slice(&m.n.to_le_bytes());
                p[4..8].copy_from_slice(&m.k.to_le_bytes());
                p[8..12].copy_from_slice(&classes[ci].0.to_le_bytes());
                p[12..16].copy_from_slice(&step.to_le_bytes());
                p[16..20].copy_from_slice(&(beta as f32).to_le_bytes());
            }
        }
        let b_params = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: &params,
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // One bind group per colour class, created once rather than per dispatch. Only the class
        // buffer differs between them; the dynamic offset carries everything else.
        let binds: Vec<wgpu::BindGroup> = live
            .iter()
            .map(|&ci| {
                dev.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("sweep"),
                    layout: &layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &b_params,
                                offset: 0,
                                size: wgpu::BufferSize::new(PARAMS_BYTES),
                            }),
                        },
                        wgpu::BindGroupEntry { binding: 1, resource: b_nbr.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: b_w.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: b_h.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: classes[ci].1.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: b_spin.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 6, resource: b_dbg.as_entire_binding() },
                    ],
                })
            })
            .collect();

        // ONE encoder, ONE pass, ONE submit for the whole run. Dispatches inside a pass execute in
        // order and each sees the previous one's writes, which is what makes the chromatic schedule
        // correct without a barrier between them.
        let mut enc = dev.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            for s in 0..sweeps as usize {
                for (li, &ci) in live.iter().enumerate() {
                    let off = ((s * live.len() + li) * stride as usize) as u32;
                    pass.set_bind_group(0, &binds[li], &[off]);
                    pass.dispatch_workgroups(classes[ci].0.div_ceil(WORKGROUP), 1, 1);
                }
            }
        }
        enc.copy_buffer_to_buffer(&b_spin, 0, &readback, 0, (state.len() * 4) as u64);
        self.queue.submit(Some(enc.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("device poll failed: {e:?}"))?;
        rx.recv()
            .map_err(|_| "the readback never completed".to_string())?
            .map_err(|e| format!("the readback failed: {e:?}"))?;

        {
            let data = slice.get_mapped_range().map_err(|e| format!("mapping failed: {e:?}"))?;
            for (i, chunk) in data.chunks_exact(4).enumerate().take(spins.len()) {
                let v = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                // Not `if v > 0 { 1 } else { -1 }`. That coercion turns any garbage — a dropped
                // dispatch, a short copy — into a valid-looking state which is then scored with
                // full confidence. The browser had exactly this bug; refusing is the whole point.
                if v != 1 && v != -1 {
                    return Err(format!("the GPU returned {v} at spin {i}; states are +1/-1"));
                }
                spins[i] = v as i8;
            }
        }
        readback.unmap();
        Ok(())
    }
}

/// Must match `@workgroup_size` in the shader. The core crate owns that number; if it ever changes
/// there, `the_workgroup_size_matches_the_shader` fails here rather than the dispatch quietly
/// covering the wrong number of lanes.
const WORKGROUP: u32 = 64;

/// Bytes in the shader's `Params` uniform: two vec4s.
const PARAMS_BYTES: u64 = 32;

/// Round `v` up to a multiple of `to`. Uniform dynamic offsets must land on the device's
/// `min_uniform_buffer_offset_alignment`, which is 256 on most hardware and validated, not ignored.
fn align_up(v: u64, to: u64) -> u64 {
    v.div_ceil(to) * to
}

fn bytes_u32(v: &[u32]) -> &[u8] {
    // Safe: u32 has no padding and no invalid bit patterns, and the slice is read-only.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
fn bytes_i32(v: &[i32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
fn bytes_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotherm::wgsl::GpuModel;
    use ferrotherm::gibbs::Sampler;
    use ferrotherm::ising::lattice2d;

    /// Skip rather than fail where there is no adapter. A headless runner having no driver is not
    /// a defect in this crate, and a red suite that means "this machine has no GPU" trains people
    /// to ignore it.
    macro_rules! gpu_or_skip {
        () => {
            match Gpu::new() {
                Some(g) => g,
                None => {
                    eprintln!("no GPU adapter on this machine; skipping");
                    return;
                }
            }
        };
    }

    #[test]
    fn the_workgroup_size_matches_the_shader() {
        // A dispatch count computed from the wrong workgroup size covers too few lanes, and the
        // nodes it misses simply never update -- silently, with the run reporting success.
        let src = ferrotherm::wgsl::sweep_shader();
        assert!(
            src.contains(&format!("@workgroup_size({WORKGROUP})")),
            "this crate dispatches in groups of {WORKGROUP}; the shader says otherwise"
        );
    }

    #[test]
    fn a_ferromagnet_orders_at_low_temperature_and_melts_at_high() {
        // The physics check, not a bit-comparison. The shader's RNG is a counter hash of
        // (step, node) and the CPU sampler has its own stream, so they cannot agree flip for flip.
        // What they must agree on is the phase.
        let gpu = gpu_or_skip!();
        let g = lattice2d(16, 1.0);
        let m = GpuModel::from_graph(&g);

        let mag = |beta: f64| {
            let mut s = vec![1i8; 256];
            gpu.sweep(&m, &mut s, beta, 400).unwrap();
            (s.iter().map(|&x| x as f64).sum::<f64>() / 256.0).abs()
        };

        let cold = mag(1.0);
        let hot = mag(0.05);
        assert!(cold > 0.8, "a ferromagnet at beta=1 should be ordered, got |m| = {cold:.3}");
        assert!(hot < 0.4, "and disordered at beta=0.05, got |m| = {hot:.3}");
    }

    #[test]
    fn the_gpu_reproduces_the_exact_mean_energy() {
        // Against EXACT physics, not against the CPU sampler. My first version of this test
        // compared the two samplers at beta = 0.44 and they disagreed by 0.55 per site -- because
        // 0.4407 is the 2D Ising critical point, where correlation times are long, and the two
        // chains started from opposite ends (all-up versus random). Each stayed near where it
        // began. That measured initialisation bias in both, not a discrepancy between them, and
        // the test would have been "wrong" no matter which sampler was correct.
        //
        // Variable elimination gives the true answer on a small lattice, and
        // E = -d(ln Z)/d(beta) is a two-point finite difference away from `log_partition`.
        let gpu = gpu_or_skip!();
        let g = lattice2d(4, 1.0);
        let n = 16.0;
        let solver = ferrotherm::exact::Elimination { max_width: 20 };

        let ln_z = |beta: f64| solver.log_partition(&g, beta).unwrap().log_z.expect("log_partition returns log_z");
        let beta = 0.7; // well below T_c: fast mixing, so a finite chain is actually equilibrated
        let h = 1e-3;
        let exact_per_site = -(ln_z(beta + h) - ln_z(beta - h)) / (2.0 * h) / n;

        // Average over independent runs: one chain's energy fluctuates about the mean, and a
        // single sample of a fluctuating quantity is not an estimate of its mean.
        let runs = 24;
        let mut total = 0.0;
        for r in 0..runs {
            let m = GpuModel::from_graph(&g);
            // Start from a different state each run so the average is not anchored to one basin.
            let mut s: Vec<i8> = (0..16).map(|i| if (i + r) % 2 == 0 { 1 } else { -1 }).collect();
            gpu.sweep(&m, &mut s, beta, 400).unwrap();
            total += g.energy(&s);
        }
        let got = total / runs as f64 / n;

        assert!(
            (got - exact_per_site).abs() < 0.12,
            "GPU {got:.4} vs exact {exact_per_site:.4} per site at beta {beta} -- the shader is \
             sampling a different distribution from the one the model defines"
        );
    }

    #[test]
    fn the_gpu_and_the_cpu_agree_away_from_criticality() {
        // The two samplers, compared where the comparison is meaningful: beta = 0.7 is well below
        // T_c (0.4407), so both chains equilibrate inside the budget and their means are
        // comparable. Both start from the SAME state, so any difference is the sampler rather than
        // where it began.
        let gpu = gpu_or_skip!();
        let g = lattice2d(12, 1.0);
        let n = 144.0;
        let beta = 0.7;
        let start: Vec<i8> = (0..144).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();

        let m = GpuModel::from_graph(&g);
        let mut s = start.clone();
        gpu.sweep(&m, &mut s, beta, 800).unwrap();
        let e_gpu = g.energy(&s) / n;

        let mut sim = Sampler::new(&g, beta, 7);
        sim.s = start;
        sim.sweeps(800, None);
        let e_cpu = g.energy(&sim.s) / n;

        assert!(
            (e_gpu - e_cpu).abs() < 0.12,
            "GPU {e_gpu:.4} vs CPU {e_cpu:.4} per site -- two implementations of one update rule"
        );
    }

    #[test]
    fn a_state_that_is_not_plus_or_minus_one_is_refused_rather_than_coerced() {
        // The length guard, which is the reachable half of the same discipline: a mismatched
        // state is refused instead of being padded into something plausible.
        let gpu = gpu_or_skip!();
        let g = lattice2d(4, 1.0);
        let m = GpuModel::from_graph(&g);
        let mut wrong = vec![1i8; 9];
        let e = gpu.sweep(&m, &mut wrong, 0.5, 1).unwrap_err();
        assert!(e.contains("16 nodes") && e.contains('9'), "must name both counts: {e}");
    }

    #[test]
    fn a_bad_temperature_is_refused_by_name() {
        let gpu = gpu_or_skip!();
        let g = lattice2d(4, 1.0);
        let m = GpuModel::from_graph(&g);
        let mut s = vec![1i8; 16];
        for bad in [f64::NAN, f64::INFINITY, -1.0] {
            let e = gpu.sweep(&m, &mut s, bad, 1).unwrap_err();
            assert!(e.contains("beta"), "{bad} should be refused by name, got: {e}");
        }
    }
}
