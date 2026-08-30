//! Chromatic block-Gibbs sampling.
//!
//! One sweep updates every node exactly once, color class by color class. Within a class, node
//! updates are conditionally independent (no two adjacent) — the parallelism a TSU exploits in
//! physics and a GPU exploits in threads; here the classes are simple loops, kept in the same
//! order so CPU, WebGPU, and device runs are cross-checkable draw for draw.

use crate::graph::Graph;
use crate::ledger::Ledger;
use crate::rng::Pcg;

// The logistic the module docs write as sigma(2 beta f_i). Only the conditional test
// exercises it, so it is dead in a non-test build — kept because it is the executable
// statement of the formula graph.rs and kernel.rs both cite in prose.
#[cfg_attr(not(test), allow(dead_code))]
#[inline]
fn sigma(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub struct Sampler<'g> {
    pub g: &'g Graph,
    pub beta: f64,
    pub s: Vec<i8>,
    pub rng: Pcg,
    /// How many threads the last parallel sweep actually used. See [`Sampler::threads_used`].
    threads_used: usize,
    /// Nodes whose value is held fixed (conditioning / "clamping"); sweeps skip them.
    pub clamped: Vec<bool>,
    /// Base seed for the parallel path's per-(sweep, class, chunk) RNG streams.
    ///
    /// Kept on every target, including wasm32 where [`Sampler::sweep_par`] is the serial twin and
    /// nothing reads it. Making the FIELD conditional would make the struct's shape depend on the
    /// target, and the two things that then differ are exactly the things this crate promises do
    /// not: a `Sampler` built the same way would not be the same object across a build boundary.
    /// The dead-code allowance is the cheaper of the two, and it is scoped to the one target where
    /// the field is genuinely unread.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    par_seed: u64,
    /// Sweeps completed via the parallel path (advances its stream derivation).
    par_sweeps: u64,
}

/// The fewest nodes a thread must be given before it is worth a barrier.
///
/// # This is a guard, not a tuned optimum
///
/// The STRUCTURAL fact is fabric-independent and is the reason the constant exists at all: creating
/// an OS thread costs microseconds on every platform anyone runs this on, and a colour-class chunk
/// of a few dozen nodes costs less than that. Handing a thread less work than the synchronisation
/// around it costs is a loss on any machine, and a caller asking for threads cannot know in advance
/// that they are in that regime. Refusing to spread work that thin is the fix; it does not depend
/// on which machine is running.
///
/// The NUMBER is a heuristic. 1024 was chosen from throughput ratios on one developer laptop, and a
/// different fabric — more cores, a different memory system, a GPU-hosted queue, a many-core server
/// — will have a different crossover. It is deliberately set where the parallel path is never a
/// loss rather than where it is fastest, because the property worth guaranteeing is "asking for
/// threads cannot hurt you", and that one survives being wrong about the exact crossover in a way
/// that a performance-tuned value would not.
///
/// **So do not read this as a benchmark result.** A caller who knows their fabric should pass a
/// thread count that suits it; this only bounds that count from above, and reports what it did
/// through [`Sampler::threads_used`].
///
/// Public because a caller who wants the parallel path to engage needs to know what it is waiting
/// for: `threads_used` reporting 1 otherwise looks like a bug, and the answer is that the SMALLEST
/// colour class did not have this many nodes per thread.
pub const MIN_CHUNK: usize = 1024;

impl<'g> Sampler<'g> {
    pub fn new(g: &'g Graph, beta: f64, seed: u64) -> Self {
        let mut rng = Pcg::new(seed, 0x5EED);
        let s = (0..g.n).map(|_| rng.spin(0.5)).collect();
        Sampler {
            g,
            beta,
            s,
            rng,
            clamped: vec![false; g.n],
            par_seed: seed,
            par_sweeps: 0,
            threads_used: 1,
        }
    }

    /// Clamp node i to value v (observation / conditioning input).
    pub fn clamp(&mut self, i: usize, v: i8) {
        debug_assert!(v == 1 || v == -1);
        self.s[i] = v;
        self.clamped[i] = true;
    }

    pub fn unclamp(&mut self, i: usize) {
        self.clamped[i] = false;
    }

    /// One full chromatic sweep (every free node updated once). If a ledger is given, it is
    /// charged one Gibbs cycle per free node — the device-side price of this sweep.
    pub fn sweep(&mut self, ledger: Option<&mut Ledger>) {
        let mut updated = 0u64;
        for class in &self.g.classes {
            for &iu in class {
                let i = iu as usize;
                if self.clamped[i] {
                    continue;
                }
                let f = self.g.field(i, &self.s);
                let p_up = crate::kernel::p_up(f, self.beta);
                self.s[i] = self.rng.spin(p_up);
                updated += 1;
            }
        }
        if let Some(l) = ledger {
            l.samples += updated;
        }
    }

    /// Run `n` sweeps.
    pub fn sweeps(&mut self, n: usize, mut ledger: Option<&mut Ledger>) {
        for _ in 0..n {
            self.sweep(ledger.as_deref_mut());
        }
    }

    /// One full chromatic sweep across `threads` OS threads — the performance core.
    ///
    /// Within a color class every node's conditional is independent (no two adjacent), so the
    /// class is split into contiguous chunks, each updated by its own thread reading the shared
    /// spin field and writing only its own chunk's nodes. Reads touch only OTHER-color nodes,
    /// which no thread writes during this phase, so the access pattern is race-free by
    /// construction of the coloring.
    ///
    /// Determinism: each (sweep, class, chunk) gets its own counter-derived RNG stream, so the
    /// result is bit-reproducible for a fixed (seed, threads). A different thread count is a
    /// different, equally valid sample path (document the thread count next to the seed).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn sweep_par(&mut self, threads: usize, ledger: Option<&mut Ledger>) {
        // One sweep is a batch of one. This used to be its own implementation, with its own copy of
        // the unsafe chunk loop, and it was the worse of the two: a single sweep is exactly the
        // case where spawning a thread per colour class costs more than the class does. Deleting it
        // removed the duplicate `unsafe` block as well -- two copies of a race argument is one more
        // than anyone can keep true.
        self.sweeps_par(1, threads, ledger);
    }

    /// The wasm twin: serial, because there are no threads to spread across.
    ///
    /// `wasm32-unknown-unknown` has a std whose `thread::spawn` COMPILES and then panics at
    /// runtime with "operation not supported on this platform" — the worst of both, since nothing
    /// stops you shipping it and the failure arrives in a browser rather than in a build. So the
    /// browser gets the serial sweep, and [`Self::threads_used`] reports 1 so a caller learns what
    /// actually ran instead of trusting the number it asked for.
    ///
    /// This is not a silent downgrade. Running serially and SAYING one thread is a different thing
    /// from running serially and reporting eight.
    #[cfg(target_arch = "wasm32")]
    pub fn sweep_par(&mut self, threads: usize, ledger: Option<&mut Ledger>) {
        assert!(threads >= 1);
        self.sweep(ledger);
        self.par_sweeps += 1;
        self.threads_used = 1;
    }

    /// How many threads the last [`Self::sweep_par`] actually used, at its peak.
    ///
    /// The largest number of chunks any single colour class was split into. Not the number asked
    /// for: a class of five across four threads is a chunk of two, so three threads run. Always 1
    /// in a browser, whatever was requested. A caller that reports throughput per thread needs
    /// this number rather than the one it passed in.
    pub fn threads_used(&self) -> usize {
        self.threads_used
    }

    /// Run `n` parallel sweeps, spawning the worker threads ONCE for the whole batch.
    ///
    /// # Why this is not a loop over [`Self::sweep_par`]
    ///
    /// It was, and that made the parallel path SLOWER THAN THE SERIAL ONE at every size anyone
    /// actually runs. `sweep_par` opens a `thread::scope` per COLOUR CLASS per SWEEP, so 2,000
    /// sweeps of a two-coloured graph on eighteen threads spawned **72,000 OS threads**. Spawning
    /// costs tens of microseconds; a colour class of five hundred nodes costs a few. The work never
    /// had a chance.
    ///
    /// Measured on this machine, back to back, before and after (2D glass, 18 threads, ratio of
    /// parallel to serial throughput — above 1.0 the parallel path is winning):
    ///
    /// ```text
    ///    spins   per class   was     is    threads that now run
    ///    1,024         512  0.03x  1.00x   1  (below the floor: the serial path)
    ///    4,096       2,048  0.13x  1.24x   2
    ///    9,216       4,608     --  2.58x   4
    ///   16,384       8,192  0.50x  1.85x   8
    /// ```
    ///
    /// Below about 32,000 spins a caller who asked for eighteen threads was handed something up to
    /// **thirty-three times slower** than not asking. That is not a tuning parameter, it is a trap,
    /// and it was reachable from the C ABI as `ft_sweep_par`.
    ///
    /// **The property is the worst cell, not the best one.** A speedup that is sometimes a slowdown
    /// is a coin toss a caller cannot call. `examples/par_scaling` reports the worst and best cells
    /// together for that reason, and what [`MIN_CHUNK`] buys is the left-hand end of that column.
    ///
    /// # What is preserved exactly
    ///
    /// **Bit-identical results.** Thread `ti` takes chunk `ti` of every class exactly as before, and
    /// each (sweep, class, chunk) derives the same counter-based stream from the same seed, so this
    /// is a scheduling change and not a numerical one. `parallel_sweeps_are_bit_identical_to_the_old_
    /// spawn_per_sweep_shape` pins that against a hand-rolled reference implementing the old shape.
    ///
    /// The barrier is what makes it safe: within a colour class no two nodes are adjacent, so the
    /// chunks may run concurrently — but class `c+1` reads what class `c` wrote, so every thread
    /// must finish a class before any thread starts the next. Every worker waits at every class
    /// boundary INCLUDING the ones where it has no chunk, or the barrier count would not match and
    /// the run would deadlock.
    pub fn sweeps_par(&mut self, n: usize, threads: usize, ledger: Option<&mut Ledger>) {
        assert!(threads >= 1);
        if n == 0 {
            return;
        }
        // One thread, or a graph with nothing to split, is the serial path -- and taking it here
        // rather than spawning one worker keeps the cheap case cheap.
        if threads == 1 || self.g.classes.is_empty() {
            for _ in 0..n {
                self.sweep(None);
            }
            self.par_sweeps += n as u64;
            self.threads_used = 1;
            self.charge_par(n, ledger);
            return;
        }

        // THE FLOOR IS WHAT MAKES ASKING FOR THREADS SAFE, and the argument for having one is
        // structural rather than measured: a thread costs microseconds to create on every platform,
        // and a chunk of a few dozen nodes costs less, so spreading work thinner than the
        // synchronisation around it is a loss on any fabric. See [`MIN_CHUNK`] on why the NUMBER is
        // a guard and not a tuned optimum.
        //
        // One machine's ratios, recorded as the observation that set the guard and not as a claim
        // about any other fabric (2D glass, 18 threads, arms interleaved so a load spike hits both
        // alike -- the first attempt did not interleave and chose 256, which is a loss):
        //
        //   min chunk | n=1024   2304   4096   9216  16384
        //   ----------|-------------------------------------
        //           1 |   0.09   0.21   0.37   0.70   1.17
        //         256 |   0.48   0.61   0.66   0.71   1.17
        //  -->   1024 |   1.02   0.98   1.27   2.06   2.68
        //        4096 |   1.00   0.99   1.02   0.98   1.42
        //
        let smallest = self.g.classes.iter().map(|c| c.len()).min().unwrap_or(0);
        let threads = threads.min((smallest / MIN_CHUNK).max(1));
        if threads <= 1 {
            for _ in 0..n { self.sweep(None); }
            self.par_sweeps += n as u64;
            self.threads_used = 1;
            self.charge_par(n, ledger);
            return;
        }

        let g = self.g;
        let beta = self.beta;
        let base = self.par_seed;
        let first_sweep = self.par_sweeps;

        // The chunk layout is fixed for the whole batch, which is what lets thread `ti` keep taking
        // chunk `ti` and so keeps the RNG streams identical to the old shape.
        let layout: Vec<(usize, usize)> = g
            .classes
            .iter()
            .map(|c| {
                let chunk = c.len().div_ceil(threads);
                let parts = if chunk == 0 { 0 } else { c.len().div_ceil(chunk) };
                (chunk, parts)
            })
            .collect();
        let workers = layout.iter().map(|&(_, parts)| parts).max().unwrap_or(1).max(1);

        let sp = self.s.as_mut_ptr() as usize;
        let clamped = &self.clamped;
        let barrier = std::sync::Barrier::new(workers);

        std::thread::scope(|scope| {
            for ti in 0..workers {
                let barrier = &barrier;
                let layout = &layout;
                scope.spawn(move || {
                    let s_ptr = sp as *mut i8;
                    for sweep in 0..n {
                        let sweep_idx = first_sweep + sweep as u64;
                        for (ci, class) in g.classes.iter().enumerate() {
                            let (chunk, parts) = layout[ci];
                            if ti < parts {
                                // SAFETY: chunks are disjoint index sets within one colour class,
                                // so every write target is unique to one thread; every read is a
                                // bias, an other-colour neighbour (not written during this class),
                                // or this thread's own not-yet-updated node. The barrier below is
                                // what makes the second of those true across classes.
                                let lo = ti * chunk;
                                let hi = (lo + chunk).min(class.len());
                                let mut rng = Pcg::new(
                                    base ^ sweep_idx.wrapping_mul(0x9E3779B97F4A7C15)
                                        ^ (ci as u64) << 32,
                                    0xC0DE ^ ti as u64,
                                );
                                for &iu in &class[lo..hi] {
                                    let i = iu as usize;
                                    if clamped[i] {
                                        continue;
                                    }
                                    let mut f = g.h[i];
                                    for k in g.offset[i]..g.offset[i + 1] {
                                        f += g.w[k]
                                            * unsafe { *s_ptr.add(g.nbr[k] as usize) } as f64;
                                    }
                                    let p_up = crate::kernel::p_up(f, beta);
                                    unsafe {
                                        *s_ptr.add(i) = rng.spin(p_up);
                                    }
                                }
                            }
                            // EVERY worker waits, including one with no chunk in this class: the
                            // barrier's participant count is fixed at `workers`, and a thread that
                            // skipped a wait would hang the rest of the batch.
                            barrier.wait();
                        }
                    }
                });
            }
        });

        self.par_sweeps += n as u64;
        self.threads_used = workers;
        self.charge_par(n, ledger);
    }

    /// Bill `n` parallel sweeps to the ledger: one sample per free node per sweep.
    fn charge_par(&self, n: usize, ledger: Option<&mut Ledger>) {
        if let Some(l) = ledger {
            let free = self.clamped.iter().filter(|&&c| !c).count() as u64;
            l.samples += free * n as u64;
        }
    }

    /// Read the full state (device price: one read per node). Prefer [`Self::read_subset`]:
    /// full-state readback is the crossings-tax regime.
    pub fn read_all(&self, ledger: Option<&mut Ledger>) -> Vec<i8> {
        if let Some(l) = ledger {
            l.reads += self.g.n as u64;
        }
        self.s.clone()
    }

    /// Read only the named nodes (e.g. action bits).
    pub fn read_subset(&self, idx: &[usize], ledger: Option<&mut Ledger>) -> Vec<i8> {
        if let Some(l) = ledger {
            l.reads += idx.len() as u64;
        }
        idx.iter().map(|&i| self.s[i]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::sigma;
    use super::*;
    use crate::graph::GraphBuilder;

    /// The sampler's stationary distribution must match the exact Boltzmann distribution on an
    /// enumerable system. 4-node cycle, mixed couplings and biases, TV < 0.02.
    #[test]
    fn matches_exact_boltzmann() {
        let mut gb = GraphBuilder::new(4);
        gb.couple(0, 1, 0.7);
        gb.couple(1, 2, -0.4);
        gb.couple(2, 3, 0.55);
        gb.couple(3, 0, 0.3);
        gb.bias(0, 0.2);
        gb.bias(2, -0.35);
        let g = gb.build();
        let beta = 0.9;

        // exact
        let mut z = 0.0;
        let mut p_exact = [0.0f64; 16];
        for m in 0..16u32 {
            let s: Vec<i8> = (0..4).map(|b| if m >> b & 1 == 1 { 1 } else { -1 }).collect();
            let w = (-beta * g.energy(&s)).exp();
            p_exact[m as usize] = w;
            z += w;
        }
        for p in p_exact.iter_mut() {
            *p /= z;
        }

        // sampled
        let mut smp = Sampler::new(&g, beta, 0xC0FFEE);
        smp.sweeps(200, None); // burn-in
        let mut counts = [0u64; 16];
        let n_samples = 200_000;
        for _ in 0..n_samples {
            smp.sweep(None);
            let mut m = 0usize;
            for b in 0..4 {
                if smp.s[b] == 1 {
                    m |= 1 << b;
                }
            }
            counts[m] += 1;
        }
        let tv: f64 = (0..16)
            .map(|m| (counts[m] as f64 / n_samples as f64 - p_exact[m]).abs())
            .sum::<f64>()
            / 2.0;
        assert!(tv < 0.02, "TV distance to exact Boltzmann = {tv}");
    }

    /// The parallel path must satisfy the same physics standard as the sequential one: Onsager's
    /// exact magnetization on the 2D lattice, and bit-reproducibility for fixed (seed, threads).
    #[test]
    fn parallel_sweep_physics_and_determinism() {
        let g = crate::ising::lattice2d(48, 1.0);
        let beta = 0.6;
        let mut smp = Sampler::new(&g, beta, 0x9A7);
        for s in smp.s.iter_mut() {
            *s = 1;
        }
        smp.sweeps_par(2000, 8, None);
        let mut acc = 0.0;
        let reads = 2000;
        for _ in 0..reads {
            smp.sweep_par(8, None);
            let m: i64 = smp.s.iter().map(|&v| v as i64).sum();
            acc += (m as f64 / g.n as f64).abs();
        }
        let m = acc / reads as f64;
        let exact = crate::ising::onsager_m(beta);
        assert!((m - exact).abs() < 0.01, "parallel |M| {m:.4} vs Onsager {exact:.4}");
        // determinism for fixed (seed, threads)
        let mut a = Sampler::new(&g, beta, 0x1234);
        let mut b = Sampler::new(&g, beta, 0x1234);
        a.sweeps_par(50, 4, None);
        b.sweeps_par(50, 4, None);
        assert_eq!(a.s, b.s, "same (seed, threads) must reproduce bit-identically");
    }

    /// Clamped nodes must never change and must steer the conditional distribution.
    #[test]
    fn clamping_conditions() {
        let mut gb = GraphBuilder::new(2);
        gb.couple(0, 1, 1.5);
        let g = gb.build();
        let mut smp = Sampler::new(&g, 1.0, 7);
        smp.clamp(0, 1);
        let mut up = 0u64;
        let n = 20_000;
        for _ in 0..n {
            smp.sweep(None);
            assert_eq!(smp.s[0], 1);
            if smp.s[1] == 1 {
                up += 1;
            }
        }
        // exact: P(s1=+1 | s0=+1) = sigma(2*beta*J) = sigma(3.0)
        let want = sigma(3.0);
        let got = up as f64 / n as f64;
        assert!((got - want).abs() < 0.01, "got {got}, want {want}");
    }
}
