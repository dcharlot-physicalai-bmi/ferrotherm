//! Packed checkerboard for an open Ising grid.
//!
//! Spins of one colour sit in `u32` words. One thread owns one word, so a site update is a
//! bit write in a register and not an atomic. The other colour is only read. The draw is
//! [`ferrotherm::grid::packed_hash`], the same function the CPU mirror runs, so a fixed seed
//! is the same spins on both.

use super::{Gpu, align_up};
use ferrotherm::grid::{packed_checkerboard_shader, packed_seed, threshold_table};
use wgpu::util::DeviceExt;

const WORKGROUP: u32 = 64;
const PARAMS_BYTES: u64 = 32;

impl Gpu {
    /// `sweeps` checkerboard updates, continuing from `start_sweep`.
    ///
    /// One pipeline, one submit. The spins are read back once at the end. No adapter is the
    /// caller's problem: this method is on a [`Gpu`] that already exists.
    ///
    /// # Errors
    ///
    /// An odd or zero side, a state whose length is not `side²`, a non-finite coupling or a
    /// negative temperature, or a GPU readback that does not return.
    pub fn sweep_packed(
        &self,
        side: usize,
        j: f64,
        beta: f64,
        spins: &mut [i8],
        sweeps: u32,
        seed: u64,
        start_sweep: u32,
    ) -> Result<(), String> {
        if side == 0 || !side.is_multiple_of(2) {
            return Err(format!(
                "packed checkerboard needs an even side, got {side}"
            ));
        }
        if spins.len() != side * side {
            return Err(format!(
                "packed checkerboard expected {} spins for side {side}, got {}",
                side * side,
                spins.len()
            ));
        }
        if !j.is_finite() || !beta.is_finite() || beta < 0.0 {
            return Err(format!(
                "packed checkerboard J {j} beta {beta} is not usable"
            ));
        }
        if sweeps == 0 {
            return Ok(());
        }
        let half = side / 2;
        let nw = half.div_ceil(32);
        let nwords_total = side * nw;
        let mut plane = [vec![0u32; nwords_total], vec![0u32; nwords_total]];
        for y in 0..side {
            for x in 0..side {
                if spins[y * side + x] > 0 {
                    let color = (x + y) & 1;
                    let k = x / 2;
                    plane[color][y * nw + k / 32] |= 1u32 << (k % 32);
                }
            }
        }
        let thresh = threshold_table(j, beta);
        let dev = &self.device;
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let b0 = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("plane0"),
            contents: bytes_u32(&plane[0]),
            usage,
        });
        let b1 = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("plane1"),
            contents: bytes_u32(&plane[1]),
            usage,
        });
        let b_thresh = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("thresh"),
            contents: bytes_u32(&thresh),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let stride = align_up(
            PARAMS_BYTES,
            dev.limits().min_uniform_buffer_offset_alignment as u64,
        );
        let steps = sweeps as usize * 2;
        let mut params = vec![0u8; steps * stride as usize];
        let seed32 = packed_seed(seed);
        for s in 0..sweeps as usize {
            for color in 0..2u32 {
                let at = (s * 2 + color as usize) * stride as usize;
                let sweep = start_sweep.wrapping_add(s as u32);
                write_params(
                    &mut params[at..at + PARAMS_BYTES as usize],
                    side as u32,
                    nw as u32,
                    color,
                    sweep,
                    seed32,
                    half as u32,
                );
            }
        }
        let b_params = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("packed-params"),
            contents: &params,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("packed-checkerboard"),
            source: wgpu::ShaderSource::Wgsl(packed_checkerboard_shader().into()),
        });
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
            label: Some("packed"),
            entries: &[
                entry(
                    0,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(PARAMS_BYTES),
                    },
                ),
                entry(1, sto(false)),
                entry(2, sto(true)),
                entry(3, sto(true)),
            ],
        });
        let pipeline_layout = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("packed"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("packed"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("sweep"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind = |mine: &wgpu::Buffer, other: &wgpu::Buffer| {
            dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("packed"),
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
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: mine.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: other.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: b_thresh.as_entire_binding(),
                    },
                ],
            })
        };
        let groups = [bind(&b0, &b1), bind(&b1, &b0)];
        let read0 = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read0"),
            size: (nwords_total * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let read1 = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read1"),
            size: (nwords_total * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let groups_x = (nwords_total as u32).div_ceil(WORKGROUP);
        let mut enc = dev.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            for s in 0..sweeps as usize {
                for color in 0..2usize {
                    let off = ((s * 2 + color) * stride as usize) as u32;
                    pass.set_bind_group(0, &groups[color], &[off]);
                    pass.dispatch_workgroups(groups_x, 1, 1);
                }
            }
        }
        let bytes = (nwords_total * 4) as u64;
        enc.copy_buffer_to_buffer(&b0, 0, &read0, 0, bytes);
        enc.copy_buffer_to_buffer(&b1, 0, &read1, 0, bytes);
        self.queue.submit(Some(enc.finish()));
        let w0 = map_u32(&self.device, &read0)?;
        let w1 = map_u32(&self.device, &read1)?;
        if w0.len() < nwords_total || w1.len() < nwords_total {
            return Err(format!(
                "packed readback was {} and {} words, expected {nwords_total}",
                w0.len(),
                w1.len()
            ));
        }
        for y in 0..side {
            for x in 0..side {
                let color = (x + y) & 1;
                let k = x / 2;
                let words = if color == 0 { &w0 } else { &w1 };
                let word = words[y * nw + k / 32];
                spins[y * side + x] = if (word >> (k % 32)) & 1 == 1 { 1 } else { -1 };
            }
        }
        Ok(())
    }
}

fn write_params(
    p: &mut [u8],
    side: u32,
    nwords: u32,
    color: u32,
    sweep: u32,
    seed: u32,
    half: u32,
) {
    p[0..4].copy_from_slice(&side.to_le_bytes());
    p[4..8].copy_from_slice(&nwords.to_le_bytes());
    p[8..12].copy_from_slice(&color.to_le_bytes());
    p[12..16].copy_from_slice(&sweep.to_le_bytes());
    p[16..20].copy_from_slice(&seed.to_le_bytes());
    p[20..24].copy_from_slice(&half.to_le_bytes());
}

fn bytes_u32(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn map_u32(device: &wgpu::Device, buf: &wgpu::Buffer) -> Result<Vec<u32>, String> {
    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| format!("device poll failed: {e:?}"))?;
    rx.recv()
        .map_err(|_| "the readback never completed".to_string())?
        .map_err(|e| format!("the readback failed: {e:?}"))?;
    let data = slice
        .get_mapped_range()
        .map_err(|e| format!("mapping failed: {e:?}"))?;
    let mut out = Vec::with_capacity(data.len() / 4);
    for chunk in data.chunks_exact(4) {
        out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    drop(data);
    buf.unmap();
    Ok(out)
}
