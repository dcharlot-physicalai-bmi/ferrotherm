//! A dynamical-system instruction set, lowered onto this crate — and therefore priced by it.
//!
//! # What this is
//!
//! The coupled-oscillator programme published an instruction set for "dynamical system units":
//! nine instructions in five phases — **connect** the fabric, **load** initial states and
//! parameters, **lock** boundary components so their physical state is held, **evolve** everything
//! unlocked at once, **store** the result — with a *label-and-trigger* discipline in which the lock
//! masks are set first and a single evolution instruction releases the dynamics simultaneously.
//! Multi-stage computation is a cascade: the equilibrium one stage reaches becomes the physical
//! input driving the next.
//!
//! That is an abstraction this crate already implements, under other names, on two substrates. This
//! module says so in executable form. [`Insn`] is the instruction set; [`Program::assemble`]
//! separates configuration from runtime the way real hardware does; and the same assembled program
//! runs on [`Substrate::Spins`]' Gibbs sampler and on [`Substrate::Phases`]' oscillator integrator
//! through one trait, which is the property this crate already claims for CPUs and fabrics.
//!
//! # The lowering, in full
//!
//! | DS-ISA | ferrotherm | Ledger |
//! |---|---|---|
//! | `Connect i j w` | [`crate::graph::GraphBuilder::couple`] | one write per endpoint |
//! | `Disconnect i j` | the coupling is dropped before `build` | one write per endpoint |
//! | `Bias i h` | [`crate::graph::GraphBuilder::bias`] | one write |
//! | `Load i v` | the sampler's initial state at `i` | one write |
//! | `Lock i v` | [`crate::gibbs::Sampler::clamp`], deferred to the next `Evolve` | one write |
//! | `Unlock i` | [`crate::gibbs::Sampler::unclamp`], likewise deferred | one write |
//! | `Evolve t` | [`crate::gibbs::Sampler::sweeps`] or [`crate::kuramoto::Kuramoto::step_euler`] | one sample per node per step |
//! | `Store i` | [`crate::gibbs::Sampler::read_subset`] | one read per node stored |
//! | `Barrier` | a synchronisation point; no device operation | nothing |
//!
//! Nine instructions, nine rows, and every row ends in a ledger charge. That last column is the
//! point of the module: **an instruction set that cannot price its own programs is not yet a cost
//! model**, and the published one states no energy for any instruction. Lowered here, every DS-ISA
//! program acquires a joules figure on any device model in [`crate::ledger`] — including the
//! oscillator floor in [`crate::precision::oscillator_fabric_floor`], whose read price is a bound no
//! analogue machine can undercut.
//!
//! # What the lowering exposes
//!
//! Two things, both structural rather than rhetorical.
//!
//! **Connect is a write, and writes are the expensive operation.** On the one device model in this
//! field with published per-operation energies, a write costs about 21,700 sampling updates. A
//! programming model whose first phase is "dynamically configure system connectivity to allocate
//! resources" is a programming model that spends its budget before the dynamics begin, and
//! [`Assembled::configuration_share`] computes exactly how much for a given program. The
//! label-and-trigger discipline is the right instinct — batch the configuration, release once — and
//! it is what this crate's schedules already do.
//!
//! **Evolve has no temperature, and therefore no invariant measure.** The published instruction
//! triggers deterministic collective evolution to equilibrium. In this crate the same instruction
//! carries a `beta`, which is what makes the result a *distribution* with a certificate rather than
//! a point. A caller that sets `beta = infinity` gets their semantics back — the zero-temperature
//! limit, which is relaxation to a local minimum — and
//! `the_zero_temperature_limit_reproduces_relaxation` shows it is a special case rather than a
//! rival. That is the whole absorption argument in one instruction.

use crate::gibbs::Sampler;
use crate::graph::{Graph, GraphBuilder};
use crate::kuramoto::Kuramoto;
use crate::ledger::{Ledger, Prices};

/// The instruction set.
///
/// Nine variants, matching the published five-phase organisation: configuration (`Connect`,
/// `Disconnect`, `Bias`, `Load`), boundary control (`Lock`, `Unlock`), execution (`Evolve`), readout
/// (`Store`) and synchronisation (`Barrier`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Insn {
    /// Couple two components with weight `w`.
    Connect {
        /// First endpoint.
        i: usize,
        /// Second endpoint.
        j: usize,
        /// Coupling weight.
        w: f64,
    },
    /// Remove a coupling set earlier in the same program.
    Disconnect {
        /// First endpoint.
        i: usize,
        /// Second endpoint.
        j: usize,
    },
    /// Set a component's local bias.
    Bias {
        /// Component.
        i: usize,
        /// Bias value.
        h: f64,
    },
    /// Set a component's initial state.
    Load {
        /// Component.
        i: usize,
        /// Initial value; its sign is the spin, its value the phase.
        v: f64,
    },
    /// Clamp a component so evolution cannot move it. Deferred to the next `Evolve`.
    Lock {
        /// Component.
        i: usize,
        /// Held value.
        v: f64,
    },
    /// Release a clamp. Deferred to the next `Evolve`.
    Unlock {
        /// Component.
        i: usize,
    },
    /// Release the dynamics for `steps` units of evolution — the trigger.
    Evolve {
        /// Sweeps, or integration steps.
        steps: usize,
    },
    /// Read a component's state out.
    Store {
        /// Component.
        i: usize,
    },
    /// A synchronisation point. Costs nothing and moves nothing; it exists so that a cascade can be
    /// written down as one program.
    Barrier,
}

/// What can be wrong with a program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// An instruction named a component the machine does not have.
    OutOfRange {
        /// The index named.
        i: usize,
        /// The component count.
        n: usize,
    },
    /// A self-coupling was requested.
    SelfCoupling {
        /// The index named twice.
        i: usize,
    },
    /// The program evolves nothing, so its result is its input.
    NoEvolution,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::OutOfRange { i, n } => {
                write!(f, "component {i} is past the end of a {n}-component machine")
            }
            Error::SelfCoupling { i } => write!(f, "component {i} cannot be coupled to itself"),
            Error::NoEvolution => write!(
                f,
                "the program contains no Evolve, so its store values are its load values and no \
                 dynamics were exercised"
            ),
        }
    }
}

impl core::error::Error for Error {}

/// Which physical substrate an assembled program runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Substrate {
    /// Stochastic binary components, evolved by chromatic block-Gibbs at a stated temperature.
    Spins,
    /// Continuous phase components, evolved by explicit Euler on the coupled-oscillator drift.
    Phases,
}

/// A program: instructions in order.
#[derive(Clone, Debug, Default)]
pub struct Program {
    n: usize,
    insns: Vec<Insn>,
}

impl Program {
    /// An empty program over `n` components.
    #[must_use]
    pub fn new(n: usize) -> Program {
        Program { n, insns: Vec::new() }
    }

    /// Append an instruction.
    pub fn push(&mut self, insn: Insn) -> &mut Program {
        self.insns.push(insn);
        self
    }

    /// The instructions, in order.
    #[must_use]
    pub fn insns(&self) -> &[Insn] {
        &self.insns
    }

    /// Component count.
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Split configuration from runtime, exactly as hardware does.
    ///
    /// `Connect`, `Disconnect`, `Bias` and `Load` describe a machine; everything else acts on one.
    /// Assembly resolves the first group into a [`Graph`] and an initial state, counts what
    /// configuring it costs in device writes, and leaves the rest as a runtime instruction stream.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfRange`], [`Error::SelfCoupling`] or [`Error::NoEvolution`] as each describes.
    pub fn assemble(&self) -> Result<Assembled, Error> {
        let n = self.n;
        let mut couplings: Vec<(usize, usize, f64)> = Vec::new();
        let mut bias = vec![0.0f64; n];
        let mut init = vec![0.0f64; n];
        let mut writes: u64 = 0;
        let mut runtime: Vec<Insn> = Vec::new();
        let mut evolves = 0usize;

        for &insn in &self.insns {
            match insn {
                Insn::Connect { i, j, w } => {
                    check(i, n)?;
                    check(j, n)?;
                    if i == j {
                        return Err(Error::SelfCoupling { i });
                    }
                    let (a, b) = if i < j { (i, j) } else { (j, i) };
                    if let Some(slot) = couplings.iter_mut().find(|(x, y, _)| *x == a && *y == b) {
                        slot.2 = w;
                    } else {
                        couplings.push((a, b, w));
                    }
                    // Both endpoints are reprogrammed by one connection.
                    writes += 2;
                }
                Insn::Disconnect { i, j } => {
                    check(i, n)?;
                    check(j, n)?;
                    let (a, b) = if i < j { (i, j) } else { (j, i) };
                    couplings.retain(|(x, y, _)| !(*x == a && *y == b));
                    writes += 2;
                }
                Insn::Bias { i, h } => {
                    check(i, n)?;
                    bias[i] = h;
                    writes += 1;
                }
                Insn::Load { i, v } => {
                    check(i, n)?;
                    init[i] = v;
                    writes += 1;
                }
                Insn::Evolve { steps } => {
                    evolves += steps;
                    runtime.push(insn);
                }
                Insn::Lock { i, .. } | Insn::Unlock { i } | Insn::Store { i } => {
                    check(i, n)?;
                    runtime.push(insn);
                }
                Insn::Barrier => runtime.push(insn),
            }
        }

        if evolves == 0 {
            return Err(Error::NoEvolution);
        }

        let mut b = GraphBuilder::new(n);
        for &(i, j, w) in &couplings {
            b.couple(i, j, w);
        }
        for i in 0..n {
            if bias[i] != 0.0 {
                b.set_bias(i, bias[i]);
            }
        }
        Ok(Assembled { graph: b.build(), init, runtime, configuration_writes: writes })
    }
}

fn check(i: usize, n: usize) -> Result<(), Error> {
    if i < n { Ok(()) } else { Err(Error::OutOfRange { i, n }) }
}

/// A program with its configuration resolved: a machine, an initial state, and what remains to run.
pub struct Assembled {
    graph: Graph,
    init: Vec<f64>,
    runtime: Vec<Insn>,
    configuration_writes: u64,
}

/// What a run produced.
#[derive(Clone, Debug, PartialEq)]
pub struct Trace {
    /// Every value a `Store` retrieved, in program order.
    pub stored: Vec<f64>,
    /// The device operations the whole program performed, configuration included.
    pub ledger: Ledger,
}

impl core::fmt::Debug for Assembled {
    /// The shape, not the machine. A `Graph` is a CSR triple that prints as thousands of integers,
    /// and a debug line nobody can read is a debug line nobody reads.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Assembled")
            .field("nodes", &self.graph.n)
            .field("edges", &self.graph.n_edges)
            .field("runtime_insns", &self.runtime.len())
            .field("configuration_writes", &self.configuration_writes)
            .finish()
    }
}

impl Assembled {
    /// The machine the configuration phase described.
    #[must_use]
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Device writes the configuration phase cost.
    #[must_use]
    pub fn configuration_writes(&self) -> u64 {
        self.configuration_writes
    }

    /// The runtime instruction stream.
    #[must_use]
    pub fn runtime(&self) -> &[Insn] {
        &self.runtime
    }

    /// The fraction of a run's energy spent configuring rather than computing, under `prices`.
    ///
    /// `None` when the device states no price for an operation the program performs. On the one
    /// device model in this field with published figures a write is about 21,700 sampling updates,
    /// so a program that reconnects its fabric between short evolutions spends almost everything
    /// here — which is the structural cost of an instruction set whose first phase is connectivity.
    #[must_use]
    pub fn configuration_share(&self, prices: &Prices, substrate: Substrate, beta: f64) -> Option<f64> {
        let trace = self.run(substrate, beta, 0);
        let total = trace.ledger.joules(prices)?;
        if total <= 0.0 {
            return None;
        }
        let config = self.configuration_writes as f64 * prices.e_write;
        config.is_finite().then_some(config / total)
    }

    /// Execute on the chosen substrate.
    ///
    /// `beta` is the inverse temperature for [`Substrate::Spins`]. For [`Substrate::Phases`] it is
    /// the reciprocal of the Euler step, and that is a **convention rather than a shared quantity**:
    /// the phase lane integrates a deterministic drift and has no temperature at all, so it is not
    /// "less noisy" at a larger `beta` — it is noiseless at every `beta` and merely finer. What the
    /// two share is the direction, colder on one and finer on the other, and nothing else. A caller
    /// who needs an actual temperature on continuous state needs a Langevin integrator;
    /// [`crate::nonrev::SkewLangevin`] is one, and it carries its own `beta`.
    ///
    /// # Panics
    ///
    /// If `beta` is not positive, or not finite on the phase substrate, where an infinite `beta`
    /// would be a zero step.
    #[must_use]
    pub fn run(&self, substrate: Substrate, beta: f64, seed: u64) -> Trace {
        assert!(beta > 0.0, "beta must be positive, got {beta}");
        match substrate {
            Substrate::Spins => self.run_spins(beta, seed),
            Substrate::Phases => {
                assert!(beta.is_finite(), "an infinite beta is a zero integration step");
                self.run_phases(1.0 / beta, seed)
            }
        }
    }

    fn run_spins(&self, beta: f64, seed: u64) -> Trace {
        let mut ledger = Ledger { samples: 0, reads: 0, writes: self.configuration_writes };
        let mut smp = Sampler::new(&self.graph, beta, seed);
        for i in 0..self.graph.n {
            // The sign of the loaded value is the spin; a zero loads +1 rather than picking at
            // random, because a program's result should not depend on an unstated coin.
            smp.s[i] = if self.init[i] < 0.0 { -1 } else { 1 };
        }
        let mut pending: Vec<(usize, Option<i8>)> = Vec::new();
        let mut stored = Vec::new();
        for &insn in &self.runtime {
            match insn {
                Insn::Lock { i, v } => {
                    pending.push((i, Some(if v < 0.0 { -1 } else { 1 })));
                    ledger.writes += 1;
                }
                Insn::Unlock { i } => {
                    pending.push((i, None));
                    ledger.writes += 1;
                }
                Insn::Evolve { steps } => {
                    // Label-and-trigger: every pending mask lands at once, then the dynamics run.
                    for &(i, v) in &pending {
                        match v {
                            Some(val) => smp.clamp(i, val),
                            None => smp.unclamp(i),
                        }
                    }
                    pending.clear();
                    smp.sweeps(steps, Some(&mut ledger));
                }
                Insn::Store { i } => {
                    let v = smp.read_subset(&[i], Some(&mut ledger));
                    stored.push(f64::from(v[0]));
                }
                Insn::Barrier => {}
                Insn::Connect { .. }
                | Insn::Disconnect { .. }
                | Insn::Bias { .. }
                | Insn::Load { .. } => {
                    unreachable!("configuration instructions are resolved by assemble")
                }
            }
        }
        Trace { stored, ledger }
    }

    fn run_phases(&self, dt: f64, _seed: u64) -> Trace {
        let n = self.graph.n;
        // The same couplings, read as a symmetric oscillator network. Symmetric because a Graph is
        // symmetric, which is exactly why this lane has a Lyapunov function and the learned
        // asymmetric networks do not -- see `crate::kuramoto`.
        let mut k = vec![0.0; n * n];
        for i in 0..n {
            for idx in self.graph.offset[i]..self.graph.offset[i + 1] {
                let j = self.graph.nbr[idx] as usize;
                k[i * n + j] = self.graph.w[idx];
            }
        }
        let sys = Kuramoto::gradient(k, n).expect("a Graph yields a symmetric zero-frequency system");
        let mut ledger = Ledger { samples: 0, reads: 0, writes: self.configuration_writes };
        let mut phi: Vec<f64> = self.init.clone();
        let mut locked: Vec<Option<f64>> = vec![None; n];
        let mut pending: Vec<(usize, Option<f64>)> = Vec::new();
        let mut stored = Vec::new();
        for &insn in &self.runtime {
            match insn {
                Insn::Lock { i, v } => {
                    pending.push((i, Some(v)));
                    ledger.writes += 1;
                }
                Insn::Unlock { i } => {
                    pending.push((i, None));
                    ledger.writes += 1;
                }
                Insn::Evolve { steps } => {
                    for &(i, v) in &pending {
                        locked[i] = v;
                        if let Some(val) = v {
                            phi[i] = val;
                        }
                    }
                    pending.clear();
                    let held: Vec<(usize, f64)> =
                        (0..n).filter_map(|i| locked[i].map(|v| (i, v))).collect();
                    let moving = (n - held.len()) as u64;
                    for _ in 0..steps {
                        // The ledger is charged by hand rather than by `step_euler`, because a
                        // locked component is not updated and must not be billed as one. The spin
                        // lane's sampler already skips clamped nodes; the two lanes would otherwise
                        // report different device accounts for the same program, which is the
                        // convention drift `ledger` carries a paragraph about.
                        sys.step_euler(&mut phi, dt, None);
                        for &(i, v) in &held {
                            phi[i] = v;
                        }
                        ledger.samples += moving;
                    }
                }
                Insn::Store { i } => {
                    ledger.reads += 1;
                    stored.push(phi[i]);
                }
                Insn::Barrier => {}
                Insn::Connect { .. }
                | Insn::Disconnect { .. }
                | Insn::Bias { .. }
                | Insn::Load { .. } => {
                    unreachable!("configuration instructions are resolved by assemble")
                }
            }
        }
        Trace { stored, ledger }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::Z1_SPICE;

    /// A four-component ring with one clamped boundary, evolved and read out.
    fn ring_program() -> Program {
        let mut p = Program::new(4);
        for i in 0..4 {
            p.push(Insn::Connect { i, j: (i + 1) % 4, w: 1.0 });
        }
        p.push(Insn::Load { i: 0, v: 1.0 });
        p.push(Insn::Lock { i: 0, v: 1.0 });
        p.push(Insn::Evolve { steps: 200 });
        for i in 0..4 {
            p.push(Insn::Store { i });
        }
        p
    }

    #[test]
    fn the_lowering_is_exactly_the_hand_written_program() {
        // The equivalence claim: the ISA is not a wrapper with its own semantics, it is a spelling.
        let asm = ring_program().assemble().expect("well formed");
        let got = asm.run(Substrate::Spins, 0.7, 4242);

        // The same thing, written directly against the crate.
        let mut b = GraphBuilder::new(4);
        for i in 0..4 {
            b.couple(i, (i + 1) % 4, 1.0);
        }
        let g = b.build();
        let mut led = Ledger { samples: 0, reads: 0, writes: asm.configuration_writes() + 1 };
        let mut smp = Sampler::new(&g, 0.7, 4242);
        for i in 0..4 {
            smp.s[i] = 1;
        }
        smp.clamp(0, 1);
        smp.sweeps(200, Some(&mut led));
        let mut want = Vec::new();
        for i in 0..4 {
            want.push(f64::from(smp.read_subset(&[i], Some(&mut led))[0]));
        }

        assert_eq!(got.stored, want, "the lowered program must produce the same states");
        assert_eq!(got.ledger, led, "and the same device account, operation for operation");
    }

    #[test]
    fn the_clamped_component_is_held_through_evolution() {
        let asm = ring_program().assemble().expect("well formed");
        for seed in [1u64, 2, 3, 99] {
            let t = asm.run(Substrate::Spins, 0.9, seed);
            assert_eq!(t.stored[0], 1.0, "a locked component moved at seed {seed}");
        }
    }

    #[test]
    fn one_program_runs_on_both_substrates_through_one_interface() {
        // The "same trait as a CPU" claim, extended to their instruction set.
        let asm = ring_program().assemble().expect("well formed");
        let spins = asm.run(Substrate::Spins, 0.7, 7);
        let phases = asm.run(Substrate::Phases, 100.0, 7);
        assert_eq!(spins.stored.len(), phases.stored.len());
        // Both charge the same configuration and the same readout; only the dynamics differ in
        // kind, and both are counted in the same units.
        assert_eq!(spins.ledger.writes, phases.ledger.writes);
        assert_eq!(spins.ledger.reads, phases.ledger.reads);
        assert_eq!(spins.ledger.samples, phases.ledger.samples);
        // The phase lane holds its lock too.
        assert!((phases.stored[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn the_zero_temperature_limit_reproduces_relaxation() {
        // Their Evolve has no temperature. Ours does, and setting it to infinity recovers theirs:
        // a ferromagnetic ring pinned at +1 relaxes to all +1 and stays there.
        let asm = ring_program().assemble().expect("well formed");
        let t = asm.run(Substrate::Spins, f64::INFINITY, 11);
        assert_eq!(t.stored, vec![1.0; 4], "zero temperature should land in the ground state");
    }

    #[test]
    fn configuration_can_cost_more_than_computation() {
        // The structural finding. A program that connects a fabric and then evolves briefly spends
        // most of its energy before any dynamics happen, because a write is ~21,700 updates.
        let mut short = Program::new(4);
        for i in 0..4 {
            short.push(Insn::Connect { i, j: (i + 1) % 4, w: 1.0 });
        }
        short.push(Insn::Evolve { steps: 1 });
        short.push(Insn::Store { i: 0 });
        let asm = short.assemble().expect("well formed");
        let share = asm
            .configuration_share(&Z1_SPICE, Substrate::Spins, 1.0)
            .expect("Z1_SPICE states every price");
        assert!(share > 0.99, "configuration should dominate a one-sweep program, got {share}");

        // Evolve long enough and the balance inverts -- which is the vendor's own conclusion about
        // this hardware class, reached here from their instruction set rather than their table.
        let mut long = Program::new(4);
        for i in 0..4 {
            long.push(Insn::Connect { i, j: (i + 1) % 4, w: 1.0 });
        }
        long.push(Insn::Evolve { steps: 1_000_000 });
        long.push(Insn::Store { i: 0 });
        let asm = long.assemble().expect("well formed");
        let share = asm
            .configuration_share(&Z1_SPICE, Substrate::Spins, 1.0)
            .expect("Z1_SPICE states every price");
        // Where the crossover actually is, and it is a long way out: eight writes at 153.6 pJ is
        // 1.23 nJ, so a four-node fabric needs about 390,000 sweeps before configuration falls
        // under a tenth of the bill. That number IS the finding -- an instruction set whose first
        // phase is connectivity has to evolve for a very long time to earn it back.
        assert!(share < 0.1, "a long evolution should amortise configuration, got {share}");
    }

    #[test]
    fn a_program_that_never_evolves_is_refused() {
        let mut p = Program::new(2);
        p.push(Insn::Connect { i: 0, j: 1, w: 1.0 });
        p.push(Insn::Store { i: 0 });
        assert_eq!(p.assemble().unwrap_err(), Error::NoEvolution);
    }

    #[test]
    fn out_of_range_and_self_coupling_are_refused_by_name() {
        let mut p = Program::new(2);
        p.push(Insn::Connect { i: 0, j: 5, w: 1.0 });
        assert_eq!(p.assemble().unwrap_err(), Error::OutOfRange { i: 5, n: 2 });
        let mut p = Program::new(2);
        p.push(Insn::Connect { i: 1, j: 1, w: 1.0 });
        assert_eq!(p.assemble().unwrap_err(), Error::SelfCoupling { i: 1 });
    }

    #[test]
    fn a_disconnect_removes_the_coupling_it_names() {
        let mut p = Program::new(3);
        p.push(Insn::Connect { i: 0, j: 1, w: 1.0 });
        p.push(Insn::Connect { i: 1, j: 2, w: 1.0 });
        p.push(Insn::Disconnect { i: 1, j: 0 });
        p.push(Insn::Evolve { steps: 1 });
        let asm = p.assemble().expect("well formed");
        assert_eq!(asm.graph().n_edges, 1, "one coupling should remain");
    }
}
