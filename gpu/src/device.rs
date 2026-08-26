//! The GPU as a [`Device`], so the conformance machinery can score it.
//!
//! Until this existed, `ferrotherm`'s survey of its own capabilities carried a line it had earned:
//! *"no `impl Device` for a GPU, so `conform` cannot even score the GPU path."* The fastest sampler
//! in the stack was the one path the verification machinery could not reach — it could be run, and
//! it could not be **checked against the fabric it claims to be**.
//!
//! What that check turns out to be about is precision. The CPU sampler is `f64` throughout and the
//! shader is `f32`, because that is what WGSL storage buffers hold. Every other `Device` here
//! declares its precision honestly — D-Wave as `Unstated`, the fixed-point fabric as
//! `Fixed { bits }` — and the GPU had no declaration at all, so nothing downstream could reason
//! about it. It is [`Precision::Float`] with a 24-bit mantissa, and saying so is what lets
//! `conform` compare the two paths knowing which differences are the arithmetic and which are the
//! sampler.
//!
//! ```no_run
//! use ferrotherm::{fabric::Device, ising::lattice2d, schedule::Schedule, ftp::Program};
//! use ferrotherm_gpu::GpuDevice;
//!
//! let Some(mut dev) = GpuDevice::open() else { return };
//! let g = lattice2d(32, 1.0);
//! let p = Program::from_graph(&g, &Schedule::default());
//! assert!(dev.program(&p).is_empty());
//! let state = dev.run(&Schedule::constant(0.6, 200), 7).unwrap();
//! assert_eq!(state.len(), g.n);
//! // The writes are charged, which is the term the ledger's thesis rests on.
//! assert_eq!(dev.ledger().writes, g.n as u64);
//! ```

use ferrotherm::fabric::{Device, Fabric, Precision, Unsupported};
use ferrotherm::ftp::Program;
use ferrotherm::ledger::{Ledger, Prices};
use ferrotherm::rng::Pcg;
use ferrotherm::schedule::Schedule;
use ferrotherm::graph::Graph;
use ferrotherm::wgsl::GpuModel;

/// A [`Device`] backed by the native WGSL sampler.
///
/// Separate from [`Gpu`](crate::Gpu) rather than implemented on it, because `Gpu` is a handle to an
/// adapter and holds no problem: `Device` is a stateful loader-and-runner, and merging the two
/// would make every `Gpu` carry a program it may never be given.
pub struct GpuDevice {
    gpu: crate::Gpu,
    model: Option<GpuModel>,
    /// Kept beside the `GpuModel` so a stage's result can be SCORED without a second lowering.
    /// `GpuModel` is the device-side layout and carries no energy function.
    graph: Option<Graph>,
    state: Vec<i8>,
    ledger: Ledger,
}

impl GpuDevice {
    /// Open the default adapter, or `None` where this machine exposes none.
    ///
    /// `None` means **not found here**, never "impossible" — the same contract as
    /// [`Gpu::new`](crate::Gpu::new), and worth preserving because a headless CI runner is the
    /// common case and every test around this skips rather than fails on it.
    pub fn open() -> Option<GpuDevice> {
        Some(GpuDevice::with(crate::Gpu::new()?))
    }

    /// Wrap an adapter already opened, so a caller that enumerated once does not enumerate again.
    pub fn with(gpu: crate::Gpu) -> GpuDevice {
        GpuDevice { gpu, model: None, graph: None, state: Vec::new(), ledger: Ledger::default() }
    }

    /// What the driver reports, for a caller that needs to know whether this is real silicon.
    pub fn adapter(&self) -> &wgpu::AdapterInfo {
        self.gpu.adapter()
    }

    /// True when the adapter is hardware rather than a software rasteriser.
    #[must_use = "false means a software rasteriser, whose timings say nothing about a GPU"]
    pub fn is_hardware(&self) -> bool {
        self.gpu.is_hardware()
    }
}

impl Device for GpuDevice {
    fn fabric(&self) -> Fabric {
        // NOT Prices::UNSTATED by oversight -- by fact. A GPU vendor publishes board power, which
        // is a rate for the whole card, not an energy per spin update. `ferrotherm-meter` derives
        // the per-operation figure by measuring THIS machine, and that measured value is what
        // belongs here; a datasheet number would be a different machine's.
        let mut f = Fabric::unconstrained("gpu", Prices::UNSTATED);
        // Lowers through `Program::to_graph`, which is pairwise.
        f.max_arity = 2;
        // The declaration that was missing. WGSL storage buffers hold f32: `GpuModel::w` and
        // `GpuModel::h` are `Vec<f32>`, so every coupling and field is rounded to 24 mantissa bits
        // on the way in. The CPU path keeps f64. Neither is wrong; an undeclared difference is.
        f.coupling_precision = Precision::Float { mantissa: 24 };
        f.field_precision = Precision::Float { mantissa: 24 };
        f.unstated = &[
            "per-operation energy: GPU vendors publish board power (a rate for the whole card), \
             not joules per spin update. Measure it with ferrotherm-meter on the machine that ran \
             the work.",
        ];
        f
    }

    fn program(&mut self, p: &Program) -> Vec<Unsupported> {
        let bad = self.fabric().check(p);
        if !bad.is_empty() {
            return bad;
        }
        match p.to_graph() {
            Ok(g) => {
                self.model = Some(GpuModel::from_graph(&g));
                self.state = vec![-1; g.n];
                // The write, charged -- one node's couplings, bias and clamp state flashed. The
                // CPU device charges it and the ledger's whole thesis rests on it; a GPU device
                // that skipped it would make the GPU look free at exactly the term that is not.
                self.ledger.writes += g.n as u64;
                self.graph = Some(g);
                Vec::new()
            }
            Err(e) => vec![Unsupported::Unplaceable { detail: e.to_string() }],
        }
    }

    /// Run the schedule and return the BEST state seen, not the last one.
    ///
    /// The trait's wording says "the final state" and every implementation here returns the best:
    /// `Cpu` delegates to `tempering::anneal_scheduled`, which tracks the minimum over every sweep,
    /// and `sbm` calls the same thing a best-so-far readout. That is not pedantry about wording. An
    /// anneal's last state is wherever the coldest stage happened to stop, and this returned it --
    /// which `conform` caught the moment it could reach this path at all, scoring **-57 against
    /// variable elimination's exact -59** while the CPU on the same ladder found -59. Nothing was
    /// wrong with the sampler; it was being asked the wrong question at the end.
    ///
    /// Tracked per STAGE rather than per sweep, and that difference is real: scoring a state means
    /// reading it back off the device, so per-sweep tracking would put a round trip between every
    /// sweep and spend the throughput the GPU exists for. The conformance ladder is 80 stages, so
    /// the minimum is taken over 80 checkpoints against the CPU's 3,200.
    fn run(&mut self, schedule: &Schedule, seed: u64) -> Result<Vec<i8>, String> {
        let m = self.model.as_ref().ok_or("no program loaded")?;
        let g = self.graph.as_ref().ok_or("no program loaded")?;
        if schedule.is_empty() {
            return Err("an empty schedule runs nothing; give it at least one stage".into());
        }
        // A RUN STARTS FROM THE SEED, and does not inherit the last one's answer.
        //
        // Two differences from the reference `Cpu` device, both found by a test that could not
        // otherwise have failed. `Cpu::run` builds a fresh `Sampler::new(g, beta, seed)`, whose
        // initial state is drawn from the seed -- so it resets per run AND starts somewhere the
        // seed chose. This carried `self.state` between calls and began at all-minus-one, which
        // made a second `run` start from the first one's best: two different seeds then returned
        // the same state, because the second was handed an answer it could not improve on and
        // simply gave it back. Matching `Sampler::new` exactly also means CPU and GPU now start a
        // given seed at the SAME configuration, which is what makes the two paths comparable.
        let mut rng = Pcg::new(seed, 0x5EED);
        self.state = (0..g.n).map(|_| rng.spin(0.5)).collect();
        let mut best = self.state.clone();
        let mut best_e = g.energy(&best);
        // Stage by stage, because a schedule is a temperature ladder and the shader takes one beta
        // per dispatch. The state carries across stages, which is what makes it an anneal rather
        // than a sequence of independent runs.
        for (i, st) in schedule.stages().iter().enumerate() {
            // The seed varies per stage. Reusing one stream across stages would have every stage
            // draw the same numbers at the same nodes, which is the failure the step counter
            // already exists to prevent, one level up.
            let stage_seed = seed.wrapping_add(i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            self.gpu.sweep_seeded(m, &mut self.state, st.beta, st.sweeps as u32, stage_seed)?;
            self.ledger.samples += m.n as u64 * st.sweeps as u64;
            let e = g.energy(&self.state);
            if e < best_e {
                best_e = e;
                best = self.state.clone();
            }
        }
        self.state = best.clone();
        Ok(best)
    }

    fn ledger(&self) -> Ledger {
        self.ledger
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotherm::ising::lattice2d;

    macro_rules! dev_or_skip {
        () => {{
            // Serialised for the same reason as `ferrotherm_gpu::tests::ADAPTER`: parallel
            // Vulkan device creation segfaults the NVIDIA driver stack.
            let own = crate::tests::ADAPTER.lock().unwrap_or_else(|e| e.into_inner());
            match GpuDevice::open() {
                Some(d) => (d, own),
                None => {
                    eprintln!("no GPU adapter on this machine; skipping");
                    return;
                }
            }
        }};
    }

    #[test]
    fn conform_can_finally_score_the_gpu_path() {
        // THE POINT OF THIS MODULE. `conform::run` takes a `&mut dyn Device`, so before this impl
        // existed the fastest sampler in the stack was the one path the conformance suite could not
        // reach -- runnable, and uncheckable against the fabric it claims to be.
        let (mut d, _own) = dev_or_skip!();
        if !d.is_hardware() {
            eprintln!("software rasteriser; the physics is still checked, the timings mean nothing");
        }
        let report = ferrotherm::conform::run(&mut d);
        assert!(
            report.passed(),
            "the GPU path fails conformance:\n{report}\n{}",
            report.failures().map(|c| format!("  {} -- {}", c.name, c.detail)).collect::<Vec<_>>().join("\n")
        );
    }

    #[test]
    fn the_fabric_declares_f32_rather_than_leaving_it_unsaid() {
        // The difference that was invisible. `GpuModel::{w,h}` are `Vec<f32>`, so every coupling and
        // field is rounded to 24 mantissa bits going in while the CPU path keeps f64. Neither is
        // wrong; an undeclared difference is, because nothing downstream can then tell an
        // arithmetic gap from a sampler gap.
        let (d, _own) = dev_or_skip!();
        let f = d.fabric();
        assert_eq!(f.coupling_precision, Precision::Float { mantissa: 24 });
        assert_eq!(f.field_precision, Precision::Float { mantissa: 24 });
        assert_eq!(f.max_arity, 2, "it lowers through to_graph, which is pairwise");
        assert!(!f.prices.is_stated(), "a GPU publishes board power, not joules per spin update");
        assert!(
            f.unstated.iter().any(|u| u.contains("per-operation energy")),
            "the gap has to be named, not merely left empty: {:?}",
            f.unstated
        );
    }

    #[test]
    fn the_seed_selects_a_stream_rather_than_being_swallowed() {
        // The trait hands `run` a seed and `Gpu::sweep` had none, so the obvious implementation
        // takes the argument and drops it -- and a caller varying the seed to gauge spread gets one
        // answer every time and reads a deaf sampler as a confident one. Note that `conform`'s
        // determinism case CANNOT catch this: an ignored seed is perfectly reproducible.
        //
        // Checked at the sweep level, where it is unambiguous: identical starting state, identical
        // beta, identical sweep count, two seeds.
        let (d, _own) = dev_or_skip!();
        let g = lattice2d(24, 1.0);
        let m = GpuModel::from_graph(&g);
        let hot = 0.15; // disordered, so two streams separate immediately

        let mut a = vec![1i8; g.n];
        let mut b = vec![1i8; g.n];
        let mut a_again = vec![1i8; g.n];
        d.gpu.sweep_seeded(&m, &mut a, hot, 40, 1).unwrap();
        d.gpu.sweep_seeded(&m, &mut b, hot, 40, 2).unwrap();
        d.gpu.sweep_seeded(&m, &mut a_again, hot, 40, 1).unwrap();
        assert_ne!(a, b, "two seeds produced identical states; the seed is being ignored");
        assert_eq!(a, a_again, "the same seed must reproduce; this is a seed, not just noise");

        // And seed 0 has to leave the unseeded stream exactly where it was, or every Onsager number
        // ever taken on this shader silently moved.
        let mut viaseed = vec![1i8; g.n];
        let mut unseeded = vec![1i8; g.n];
        d.gpu.sweep_seeded(&m, &mut viaseed, hot, 40, 0).unwrap();
        d.gpu.sweep(&m, &mut unseeded, hot, 40).unwrap();
        assert_eq!(viaseed, unseeded, "seed 0 must be the stream that existed before seeding");
    }

    #[test]
    fn the_device_threads_its_seed_through_to_the_sampler() {
        // The same property one level up, where it is what the trait actually promises.
        //
        // On a FRUSTRATED instance, deliberately. A ferromagnet's all-minus-one start is already a
        // ground state, so best-so-far never leaves it and both seeds return the initial state --
        // which is correct behaviour and a completely blind test. This assertion only means
        // something where the sampler has somewhere to go.
        let (mut d, _own) = dev_or_skip!();
        let inst = ferrotherm::planted::frustrated_loops(12, 24, 5);
        let p = Program::from_graph(&inst.graph, &Schedule::default());
        assert!(d.program(&p).is_empty());

        let hot = Schedule::constant(0.25, 30);
        let a = d.run(&hot, 11).unwrap();
        let b = d.run(&hot, 22).unwrap();
        assert!(
            inst.graph.energy(&a) < 0.0 && inst.graph.energy(&b) < 0.0,
            "both runs should have moved off the initial state at all"
        );
        assert_ne!(a, b, "two seeds gave the same trajectory; the device is not threading the seed");
    }

    #[test]
    fn the_write_is_charged_because_that_is_the_term_the_ledger_rests_on() {
        let (mut d, _own) = dev_or_skip!();
        assert_eq!(d.ledger().writes, 0);
        let g = lattice2d(16, 1.0);
        let p = Program::from_graph(&g, &Schedule::default());
        assert!(d.program(&p).is_empty());
        assert_eq!(d.ledger().writes, g.n as u64, "one write per node flashed");
        assert_eq!(d.ledger().samples, 0, "loading is not sampling");

        d.run(&Schedule::constant(0.6, 10), 3).unwrap();
        assert_eq!(d.ledger().samples, g.n as u64 * 10, "one sample per node per sweep");
    }

    #[test]
    fn running_without_a_program_is_an_error_not_an_empty_state() {
        let (mut d, _own) = dev_or_skip!();
        let e = d.run(&Schedule::constant(0.6, 10), 1).unwrap_err();
        assert!(e.contains("no program"), "{e}");
        // And an empty schedule runs nothing, which is worth saying rather than returning the
        // initial state as though it had been sampled.
        let g = lattice2d(8, 1.0);
        assert!(d.program(&Program::from_graph(&g, &Schedule::default())).is_empty());
        let e2 = d.run(&Schedule::new(), 1).unwrap_err();
        assert!(e2.contains("empty schedule"), "{e2}");
    }
}
