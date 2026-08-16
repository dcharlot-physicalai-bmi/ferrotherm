# ferrotherm-gpu

Native GPU sampling for [ferrotherm](https://github.com/dcharlot-physicalai-bmi/ferrotherm): the
**same WGSL sweep the browser runs**, on Vulkan, Metal or DX12.

Kept out of the core crate so `ferrotherm` stays std-only with zero dependencies. One shader, one
set of physics, whether it runs in a browser tab or against a discrete GPU.

```sh
cargo add ferrotherm-gpu
```

## Verified on three graphics APIs

| adapter | kind | API | tests |
|---|---|---|---|
| Apple M5 Max | IntegratedGpu | Metal | 6/6 |
| NVIDIA L4 (EC2 `g6.xlarge`) | DiscreteGpu | Vulkan 1.4 | 6/6 |
| Microsoft Basic Render Driver | **Cpu** | DX12 | 6/6 |

All three reproduce the **exact mean energy computed by variable elimination**, not merely a
plausible-looking distribution. That is worth checking on each rather than assuming: a shader can
pass on Metal and fail on Vulkan, whose validation is stricter and whose f32 behaviour differs.

**The DX12 row is WARP, a software rasteriser, and that is a real limit on what it proves.** It
establishes that the shader compiles under DX12 and that the physics is right. It says nothing about
DX12 on hardware, because there was no GPU on that machine.

## It tells you what it is running on

```rust
let gpu = Gpu::new()?;
if !gpu.is_hardware() {
    // a software adapter: correctness still holds, timings mean nothing
}
```

`is_hardware()` exists because a benchmark that cannot tell a GPU from a CPU emulating one will
happily report a speedup against itself. On the WARP run above it reported `Cpu` and the bundled
benchmark declined to quote a figure.

The other number worth stating plainly: an earlier **54× vs CPU** claim here was measured against
**one CPU core**. Against `sweeps_par` on the same machine it is roughly **12× faster and 10×
cheaper per flip** — still worth having, and not what the first figure said.

Apache-2.0. From the Institute for Physical AI @ BMI.
