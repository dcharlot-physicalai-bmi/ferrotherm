//! What a fabric can do, declared — and checked before a program reaches it.
//!
//! Every open repository in this field is a simulator. Grep `thrml`, `torx`, `thermox`,
//! `posteriors`, `kaiwu-pytorch-plugin`, `SANTA` or `AOCoptimizer.jl` for
//! `pcie|usb|/dev|ioctl|fpga|driver|firmware` and you get nothing; the only open stack that drives
//! real sampling silicon belongs to D-Wave. This module is the seam that fixes that: one trait every
//! backend implements, from a CPU to a GPU to an FPGA to somebody's cloud annealer.
//!
//! # Capabilities are declared, and precision is first-class
//!
//! The motivating failure is real and recent. QBoson's coupling weights are int8, that limit is the
//! binding constraint on their entire platform, it appears nowhere in their documentation, and a
//! third party had to discover it by running experiments. A model quantised from `f64` to `int8`
//! still runs; it just answers a different question, and nothing tells you.
//!
//! So a [`Fabric`] states its limits up front — size, degree, topology, coupling and field
//! precision, whether it can hold an external field at all — and [`Fabric::check`] refuses a program
//! that exceeds them, naming the limit. Where quantisation is wanted rather than refusal,
//! [`Fabric::requantize`] performs it and **returns the error it introduced**, so the loss is a
//! number the caller has to look at rather than a silence.

use crate::ftp::Program;
use crate::ledger::Prices;

/// How a fabric's spins are wired.
#[derive(Clone, Debug, PartialEq)]
pub enum Topology {
    /// Every spin may couple to every other. Rare, and the reason it is rare is cost.
    AllToAll,
    /// A fixed maximum degree with no further structure assumed.
    Degree(usize),
    /// A named hardware graph whose structure the caller is expected to know.
    Named(&'static str),
    /// Arbitrary: a simulator, which is any backend that is not silicon.
    Unconstrained,
}

/// What a backend can actually do. Declared by the backend, checked by [`Fabric::check`].
#[derive(Clone, Debug)]
pub struct Fabric {
    pub name: &'static str,
    pub topology: Topology,
    /// Maximum spins, or `None` for "whatever fits in memory".
    pub max_spins: Option<usize>,
    /// Maximum degree, or `None` if unconstrained.
    pub max_degree: Option<usize>,
    /// Bits of coupling precision, or `None` for full `f64`.
    ///
    /// State it even when it is generous. An undeclared precision is the defect this whole module
    /// exists to prevent.
    pub coupling_bits: Option<u32>,
    /// Bits of field precision, or `None` for full `f64`.
    pub field_bits: Option<u32>,
    /// Whether an external field can be applied at all. Some fabrics cannot hold one.
    pub supports_field: bool,
    /// Maximum factor arity. Two means pairwise only, which is most hardware.
    pub max_arity: usize,
    /// What magnitudes a coupling may take, or `None` for unbounded.
    pub coupling_range: Option<Range>,
    /// What magnitudes a field may take, or `None` for unbounded.
    pub field_range: Option<Range>,
    /// Whether every coupling must have the same weight.
    ///
    /// Set by fabrics that *count* active neighbours rather than summing weighted ones. It is a
    /// severe restriction — a spin glass cannot be expressed at all — and exactly the kind of limit
    /// that goes undeclared until someone's answers come back wrong.
    pub uniform_couplings: bool,
    /// Energy prices for the ledger.
    pub prices: Prices,
}

/// The magnitudes a coefficient may take on a fabric.
///
/// Every real annealer has one and they differ in kind, not just in width. D-Wave's couplings are
/// continuous over `[-1, 1]`; Hitachi's CMOS ASIC stores four-bit integers over `-7..=7`. A program
/// with `J = 0.5` fits the first exactly and does not fit the second at all.
///
/// This is separate from `coupling_bits`, which says how finely a value is stored. A bit count
/// alone cannot distinguish `-7..=7` from a fixed-point fraction over `[-1, 1)`, and the difference
/// decides whether a program has to be requantised or merely scaled.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Range {
    pub lo: f64,
    pub hi: f64,
    /// Whether only whole numbers in the range are representable.
    pub integral: bool,
}

impl Range {
    pub const fn continuous(lo: f64, hi: f64) -> Range {
        Range { lo, hi, integral: false }
    }
    pub const fn integers(lo: f64, hi: f64) -> Range {
        Range { lo, hi, integral: true }
    }
    pub fn holds(&self, v: f64) -> bool {
        v >= self.lo && v <= self.hi && (!self.integral || v.fract() == 0.0)
    }
    /// The largest magnitude representable, which is what a scale factor is computed against.
    pub fn reach(&self) -> f64 {
        self.lo.abs().min(self.hi.abs())
    }
}

impl core::fmt::Display for Range {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.integral {
            write!(f, "the integers {}..={}", self.lo, self.hi)
        } else {
            write!(f, "[{}, {}]", self.lo, self.hi)
        }
    }
}

/// Why a program cannot run on a fabric.
#[derive(Clone, Debug, PartialEq)]
pub enum Unsupported {
    TooManySpins { need: usize, limit: usize },
    TooHighDegree { node: usize, degree: usize, limit: usize },
    ArityTooHigh { arity: usize, limit: usize },
    NoFieldSupport { nodes: usize },
    /// The program's dynamic range cannot survive the fabric's coupling precision.
    CouplingPrecision { bits: u32, worst_relative_error: f64 },
    /// The fabric counts neighbours rather than weighting them, so all couplings must be equal.
    NonUniformCouplings { distinct: usize },
    /// A coefficient outside what the fabric can represent.
    ///
    /// Often fixable: see [`Fabric::scale_to_fit`], because scaling every coefficient by one factor
    /// leaves the ground state exactly where it was.
    OutOfRange { what: &'static str, value: f64, range: Range },
}

impl core::fmt::Display for Unsupported {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Unsupported::TooManySpins { need, limit } => {
                write!(f, "the program needs {need} spins and this fabric has {limit}")
            }
            Unsupported::TooHighDegree { node, degree, limit } => write!(
                f,
                "spin {node} has degree {degree} and this fabric allows {limit}; sparsify the \
                 model or embed it before submitting"
            ),
            Unsupported::ArityTooHigh { arity, limit } => write!(
                f,
                "a factor of arity {arity} cannot run on a fabric limited to {limit}; lower it to \
                 pairwise first"
            ),
            Unsupported::NoFieldSupport { nodes } => write!(
                f,
                "{nodes} spins carry an external field and this fabric cannot apply one"
            ),
            Unsupported::OutOfRange { what, value, range } => write!(
                f,
                "a {what} of {value} is outside this fabric's {range}; scale the program to fit \
                 (Fabric::scale_to_fit) or requantise it"
            ),
            Unsupported::NonUniformCouplings { distinct } => write!(
                f,
                "this fabric counts active neighbours rather than weighting them, so every coupling \
                 must be equal; the program has {distinct} distinct weights. A spin glass cannot be \
                 expressed here at all"
            ),
            Unsupported::CouplingPrecision { bits, worst_relative_error } => write!(
                f,
                "this fabric stores couplings in {bits} bits, which would change one of them by \
                 {:.1}% -- requantize explicitly if that is acceptable, rather than discovering it \
                 from the answers",
                worst_relative_error * 100.0
            ),
        }
    }
}

impl Fabric {
    /// A simulator: no limits, full precision.
    pub fn unconstrained(name: &'static str, prices: Prices) -> Fabric {
        Fabric {
            name,
            topology: Topology::Unconstrained,
            max_spins: None,
            max_degree: None,
            coupling_bits: None,
            field_bits: None,
            supports_field: true,
            max_arity: usize::MAX,
            coupling_range: None,
            field_range: None,
            uniform_couplings: false,
            prices,
        }
    }

    /// The factor that brings a program inside this fabric's ranges, if one exists.
    ///
    /// Multiply every coefficient by the result and the program fits. This is free for
    /// **optimisation**: scaling every coupling and field by one positive number leaves the energy
    /// ordering of states untouched, so the ground state is exactly where it was.
    ///
    /// It is **not** free for sampling. The Boltzmann distribution depends on `β·E`, so scaling `E`
    /// by `s` and leaving `β` alone samples a different distribution — a hotter one for `s < 1`.
    /// Divide `β` by `s` to compensate, and check the fabric can reach that `β` at all.
    ///
    /// Returns `None` when scaling cannot help: an integral range cannot represent a program whose
    /// coefficients are not in a fixed ratio to each other, and shrinking to fit would collapse
    /// small couplings to zero.
    pub fn scale_to_fit(&self, p: &Program) -> Option<f64> {
        let worst = |r: Option<Range>, vals: &mut dyn Iterator<Item = f64>| -> Option<f64> {
            let r = r?;
            let peak = vals.map(f64::abs).fold(0.0f64, f64::max);
            if peak == 0.0 {
                return Some(f64::INFINITY); // nothing to constrain
            }
            Some(r.reach() / peak)
        };
        let a = worst(
            self.coupling_range,
            &mut p.factors.iter().map(|f| f.weight()),
        );
        let b = worst(self.field_range, &mut p.bias.iter().map(|(_, h)| *h));
        let s = match (a, b) {
            (None, None) => return None, // no ranges to satisfy
            (Some(x), None) | (None, Some(x)) => x,
            (Some(x), Some(y)) => x.min(y),
        };
        if !s.is_finite() {
            return Some(1.0); // an empty program already fits
        }
        // An integral range needs every scaled value to land on a whole number, which one factor
        // cannot generally deliver. Say so rather than returning a factor that quietly rounds.
        let integral = self.coupling_range.map(|r| r.integral).unwrap_or(false)
            || self.field_range.map(|r| r.integral).unwrap_or(false);
        if integral {
            let lands = p
                .factors
                .iter()
                .map(|f| f.weight())
                .chain(p.bias.iter().map(|(_, h)| *h))
                .all(|v| ((v * s).round() - v * s).abs() < 1e-9);
            if !lands {
                return None;
            }
        }
        Some(s)
    }

    // ---- declared fabrics ---------------------------------------------------------------------
    //
    // A fabric can be DECLARED without being reachable, and that is most of the value: a caller can
    // ask whether their program fits a machine before buying time on it, and the conformance suite
    // can score any backend that claims to be one. Every number below is from the vendor's own
    // published material, cited at the line that uses it. Where a vendor does not publish a limit,
    // the field is `None` and this file says so rather than guessing — an invented limit is worse
    // than an absent one, because it refuses programs that would have run.

    /// D-Wave Advantage2 — Zephyr-12, generally available May 2025.
    ///
    /// 4,400+ qubits and 40,000+ couplers at 20-way connectivity
    /// (<https://docs.dwavequantum.com/en/latest/quantum_research/topologies.html>, and D-Wave's
    /// general-availability announcement). Couplings are continuous over `[-1, 1]` and fields over
    /// `[-4, 4]` (`j_range`, `h_range` in the solver properties). `extended_j_range` reaches
    /// `[-2, 1]` but needs flux-bias calibration per chain, so it is not the default here.
    ///
    /// This is a quantum annealer rather than a thermodynamic sampler: it minimises an Ising energy
    /// and does not hold a temperature you set. It reads `.ftp` because the problem is the same
    /// shape, and `Certificate` does not apply to it — there is no β it claims to have sampled at.
    pub fn dwave_advantage2(prices: Prices) -> Fabric {
        Fabric {
            name: "dwave-advantage2",
            topology: Topology::Degree(20),
            max_spins: Some(4_400),
            max_degree: Some(20),
            // Analog, not quantised. The practical limit is integrated control error rather than a
            // bit count, and D-Wave publishes no bit count, so claiming one would be an invention.
            coupling_bits: None,
            field_bits: None,
            supports_field: true,
            max_arity: 2,
            coupling_range: Some(Range::continuous(-1.0, 1.0)),
            field_range: Some(Range::continuous(-4.0, 4.0)),
            uniform_couplings: false,
            prices,
        }
    }

    /// D-Wave Advantage — Pegasus, 5,640 qubits and 40,484 couplers at 15-way connectivity
    /// (<https://docs.dwavequantum.com/en/latest/quantum_research/topologies.html>). More qubits
    /// than Advantage2 and fewer couplers each, which is the trade the newer topology reverses.
    pub fn dwave_advantage(prices: Prices) -> Fabric {
        Fabric {
            name: "dwave-advantage",
            topology: Topology::Degree(15),
            max_spins: Some(5_640),
            max_degree: Some(15),
            coupling_bits: None,
            field_bits: None,
            supports_field: true,
            max_arity: 2,
            coupling_range: Some(Range::continuous(-1.0, 1.0)),
            field_range: Some(Range::continuous(-4.0, 4.0)),
            uniform_couplings: false,
            prices,
        }
    }

    /// Can this program run here as written?
    ///
    /// Returns every violation rather than the first, because a caller deciding whether to embed a
    /// model wants the whole picture in one pass.
    pub fn check(&self, p: &Program) -> Vec<Unsupported> {
        let mut out = Vec::new();

        if let Some(limit) = self.max_spins {
            if p.spins > limit {
                out.push(Unsupported::TooManySpins { need: p.spins, limit });
            }
        }

        let mut worst_arity = 0;
        let mut degree = vec![0usize; p.spins];
        for f in &p.factors {
            worst_arity = worst_arity.max(f.arity());
            if f.arity() == 2 {
                for v in f.vars() {
                    if v < p.spins {
                        degree[v] += 1;
                    }
                }
            }
        }
        if worst_arity > self.max_arity {
            out.push(Unsupported::ArityTooHigh { arity: worst_arity, limit: self.max_arity });
        }

        let deg_limit = match (&self.topology, self.max_degree) {
            (Topology::Degree(d), _) => Some(*d),
            (_, Some(d)) => Some(d),
            _ => None,
        };
        if let Some(limit) = deg_limit {
            if let Some((node, &d)) = degree.iter().enumerate().max_by_key(|(_, &d)| d) {
                if d > limit {
                    out.push(Unsupported::TooHighDegree { node, degree: d, limit });
                }
            }
        }

        if !self.supports_field && !p.bias.is_empty() {
            out.push(Unsupported::NoFieldSupport { nodes: p.bias.len() });
        }

        // Report the WORST offender rather than every one. A program built by a loop violates a
        // range in thousands of places for one reason, and a thousand identical findings buries the
        // ones that differ.
        if let Some(r) = self.coupling_range {
            if let Some(w) = p.factors.iter().map(|f| f.weight()).find(|v| !r.holds(*v)) {
                out.push(Unsupported::OutOfRange { what: "coupling", value: w, range: r });
            }
        }
        if let Some(r) = self.field_range {
            if let Some((_, h)) = p.bias.iter().find(|(_, h)| !r.holds(*h)) {
                out.push(Unsupported::OutOfRange { what: "field", value: *h, range: r });
            }
        }

        if self.uniform_couplings {
            let mut seen: Vec<u64> = p.factors.iter().map(|f| f.weight().to_bits()).collect();
            seen.sort_unstable();
            seen.dedup();
            if seen.len() > 1 {
                out.push(Unsupported::NonUniformCouplings { distinct: seen.len() });
            }
        }

        if let Some(bits) = self.coupling_bits {
            let err = Self::quantization_error(p, bits);
            // A tenth of a percent is the line: below it the model is the model, above it the
            // caller is answering a different question and should say so out loud.
            if err > 1e-3 {
                out.push(Unsupported::CouplingPrecision { bits, worst_relative_error: err });
            }
        }

        out
    }

    /// Worst relative error that quantising this program's couplings to `bits` would introduce.
    pub fn quantization_error(p: &Program, bits: u32) -> f64 {
        let max = p.factors.iter().map(|f| f.weight().abs()).fold(0.0f64, f64::max);
        if max == 0.0 || bits == 0 {
            return 0.0;
        }
        // signed, so one bit is the sign
        let levels = ((1u64 << (bits - 1)) - 1) as f64;
        let step = max / levels;
        p.factors
            .iter()
            .map(|f| {
                let w = f.weight();
                if w == 0.0 {
                    0.0
                } else {
                    ((w / step).round() * step - w).abs() / w.abs()
                }
            })
            .fold(0.0f64, f64::max)
    }

    /// Quantise a program's couplings to this fabric's precision, returning the worst relative
    /// error introduced.
    ///
    /// Explicit by design. A fabric that quantises silently is answering a different question than
    /// the one it was asked, and the caller is the last to find out.
    pub fn requantize(&self, p: &mut Program) -> f64 {
        let Some(bits) = self.coupling_bits else { return 0.0 };
        let err = Self::quantization_error(p, bits);
        let max = p.factors.iter().map(|f| f.weight().abs()).fold(0.0f64, f64::max);
        if max == 0.0 || bits == 0 {
            return 0.0;
        }
        let levels = ((1u64 << (bits - 1)) - 1) as f64;
        let step = max / levels;
        for f in &mut p.factors {
            let vars: Vec<usize> = f.vars().collect();
            let w = (f.weight() / step).round() * step;
            *f = crate::factor::Factor::new(&vars, w, p.spins).expect("requantised in place");
        }
        err
    }
}

/// A backend that can run a program.
///
/// The one seam through which every execution passes, so that adding a fabric is an implementation
/// rather than a fork.
pub trait Device {
    /// What this backend can do. Callers check against it before submitting.
    fn fabric(&self) -> Fabric;

    /// Load a program. Returns every reason it cannot run, empty on success.
    fn program(&mut self, p: &Program) -> Vec<Unsupported>;

    /// Run a schedule and return the final state.
    fn run(&mut self, schedule: &crate::schedule::Schedule, seed: u64) -> Result<Vec<i8>, String>;

    /// Operations charged so far, for the ledger.
    fn ledger(&self) -> crate::ledger::Ledger;
}

/// The reference backend: this crate's own sampler on the local CPU.
pub struct Cpu {
    graph: Option<crate::graph::Graph>,
    state: Vec<i8>,
    ledger: crate::ledger::Ledger,
}

impl Default for Cpu {
    fn default() -> Self {
        Cpu { graph: None, state: Vec::new(), ledger: crate::ledger::Ledger::default() }
    }
}

impl Device for Cpu {
    fn fabric(&self) -> Fabric {
        Fabric::unconstrained("cpu", crate::ledger::Z1_SPICE)
    }

    fn program(&mut self, p: &Program) -> Vec<Unsupported> {
        let bad = self.fabric().check(p);
        if bad.is_empty() {
            match p.to_graph() {
                Ok(g) => {
                    self.state = vec![-1; g.n];
                    self.graph = Some(g);
                }
                Err(_) => return vec![Unsupported::ArityTooHigh { arity: 3, limit: 2 }],
            }
        }
        bad
    }

    fn run(&mut self, schedule: &crate::schedule::Schedule, seed: u64) -> Result<Vec<i8>, String> {
        let g = self.graph.as_ref().ok_or("no program loaded")?;
        let (best, _) = crate::tempering::anneal_scheduled(g, schedule, seed, Some(&mut self.ledger));
        self.state = best.clone();
        Ok(best)
    }

    fn ledger(&self) -> crate::ledger::Ledger {
        self.ledger
    }
}

#[cfg(test)]
mod range_tests {
    use super::*;
    use crate::ledger::Z1_SPICE;

    fn program(weights: &[f64], fields: &[(usize, f64)]) -> Program {
        let mut src = format!("ftp 1\nspins {}\n", weights.len() + 1);
        for (i, w) in weights.iter().enumerate() {
            src.push_str(&format!("factor {w} {i} {}\n", i + 1));
        }
        for (i, h) in fields {
            src.push_str(&format!("bias {i} {h}\n"));
        }
        Program::from_ftp(&src).unwrap()
    }

    #[test]
    fn a_coupling_outside_the_range_is_named_with_the_range() {
        // D-Wave's couplings live in [-1, 1]. A program written without that in mind is the common
        // case, and "it failed" is not a useful thing to tell its author.
        let f = Fabric::dwave_advantage2(Z1_SPICE);
        let bad = f.check(&program(&[0.5, 3.0, -0.2], &[]));
        assert_eq!(bad.len(), 1, "{bad:?}");
        let msg = bad[0].to_string();
        assert!(msg.contains('3') && msg.contains("[-1, 1]"), "{msg}");

        assert!(f.check(&program(&[0.5, -1.0, 0.25], &[])).is_empty(), "these all fit");
    }

    #[test]
    fn a_field_has_its_own_wider_range() {
        // h_range is [-4, 4] where j_range is [-1, 1]; a fabric that conflated them would refuse a
        // field of 3 that the machine accepts.
        let f = Fabric::dwave_advantage2(Z1_SPICE);
        assert!(f.check(&program(&[0.5], &[(0, 3.0)])).is_empty(), "a field of 3 is fine");
        let bad = f.check(&program(&[0.5], &[(0, 9.0)]));
        assert_eq!(bad.len(), 1, "{bad:?}");
        assert!(bad[0].to_string().contains("field"), "{}", bad[0]);
    }

    #[test]
    fn an_integral_range_refuses_what_a_continuous_one_accepts() {
        // The distinction a bit count cannot make. J = 0.5 is representable on D-Wave and on no
        // machine that stores whole numbers, however many bits it has.
        let half = program(&[0.5], &[]);
        assert!(Fabric::dwave_advantage2(Z1_SPICE).check(&half).is_empty());

        let mut integral = Fabric::unconstrained("integral", Z1_SPICE);
        integral.coupling_range = Some(Range::integers(-7.0, 7.0));
        let bad = integral.check(&half);
        assert_eq!(bad.len(), 1, "{bad:?}");
        assert!(bad[0].to_string().contains("integers -7..=7"), "{}", bad[0]);
    }

    #[test]
    fn scaling_makes_a_program_fit_and_leaves_the_ground_state_where_it_was() {
        let f = Fabric::dwave_advantage2(Z1_SPICE);
        let p = program(&[2.0, -5.0, 1.0], &[(0, 8.0)]);
        assert!(!f.check(&p).is_empty(), "it does not fit as written");

        let s = f.scale_to_fit(&p).expect("scaling should help here");
        // the field is 8 against a reach of 4, the worst coupling 5 against a reach of 1: the
        // coupling binds, so 1/5
        assert!((s - 0.2).abs() < 1e-12, "{s}");

        // and the scaled program really does fit
        let scaled = program(&[2.0 * s, -5.0 * s, 1.0 * s], &[(0, 8.0 * s)]);
        assert!(f.check(&scaled).is_empty(), "{:?}", f.check(&scaled));

        // The ground state is unchanged, which is what makes scaling free for OPTIMISATION. Every
        // state's energy is multiplied by the same positive number, so their order is identical.
        let exact = crate::exact::Elimination::default();
        let a = exact.ground_state(&p.to_graph().unwrap()).unwrap();
        let b = exact.ground_state(&scaled.to_graph().unwrap()).unwrap();
        assert_eq!(a.ground_state, b.ground_state, "scaling must not move the optimum");
        let (ea, eb) = (a.ground_energy.unwrap(), b.ground_energy.unwrap());
        assert!((eb - ea * s).abs() < 1e-9, "and the energy scales exactly: {ea} {eb} {s}");
    }

    #[test]
    fn scaling_moves_the_distribution_unless_beta_compensates() {
        // The caveat on `scale_to_fit`, measured rather than asserted. Scaling is free for
        // OPTIMISATION -- the previous test shows the ground state does not move -- and it is not
        // free for SAMPLING, because the Boltzmann weight depends on the product beta*E. Scaling E
        // by s and leaving beta alone samples a hotter distribution.
        let mut b = crate::graph::GraphBuilder::new(6);
        for i in 0..5 {
            b.couple(i, i + 1, if i % 2 == 0 { 1.0 } else { -0.7 });
        }
        b.set_bias(0, 0.4);
        let g = b.build();

        let s = 0.25;
        let mut sb = crate::graph::GraphBuilder::new(g.n);
        for i in 0..g.n {
            for k in g.offset[i]..g.offset[i + 1] {
                let j = g.nbr[k] as usize;
                if i < j {
                    sb.couple(i, j, g.w[k] * s);
                }
            }
            sb.set_bias(i, g.h[i] * s);
        }
        let scaled = sb.build();

        let beta = 1.2;
        let tv = |a: &[f64], b: &[f64]| -> f64 {
            0.5 * a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>()
        };
        let want = crate::ising::exact_boltzmann(&g, beta);
        let same_beta = crate::ising::exact_boltzmann(&scaled, beta);
        let fixed_beta = crate::ising::exact_boltzmann(&scaled, beta / s);

        let drift = tv(&want, &same_beta);
        let corrected = tv(&want, &fixed_beta);
        assert!(drift > 0.2, "scaling really does move the distribution: TV {drift:.4}");
        assert!(
            corrected < 1e-12,
            "and dividing beta by the same factor puts it back exactly: TV {corrected:.2e}"
        );
    }

    #[test]
    fn scaling_is_free_for_optimisation_and_is_not_free_for_sampling() {
        // The caveat on `scale_to_fit`, measured rather than asserted. Scaling every coefficient by
        // s multiplies every state's energy by s, which leaves their ORDER alone -- so the optimum
        // does not move. The Boltzmann weight is exp(-beta*E), so the same scaling at the same beta
        // is a different distribution, and a caller who scales to fit a fabric and then samples has
        // silently asked a different question.
        let s = 0.25;
        let base = crate::ising::ring(8, 1.0, 0.3);
        let mut b = crate::graph::GraphBuilder::new(base.n);
        for i in 0..base.n {
            for k in base.offset[i]..base.offset[i + 1] {
                let j = base.nbr[k] as usize;
                if j > i {
                    b.couple(i, j, base.w[k] * s);
                }
            }
            b.set_bias(i, base.h[i] * s);
        }
        let scaled = b.build();

        let beta = 1.0;
        let exact_of = |g: &crate::graph::Graph, beta: f64| crate::ising::exact_boltzmann(g, beta);
        let tv = |a: &[f64], b: &[f64]| {
            a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>() / 2.0
        };

        let p0 = exact_of(&base, beta);
        let same_beta = exact_of(&scaled, beta);
        let compensated = exact_of(&scaled, beta / s);

        let drift = tv(&p0, &same_beta);
        let fixed = tv(&p0, &compensated);
        assert!(drift > 0.1, "scaling at a fixed beta really does move the distribution: {drift}");
        assert!(fixed < 1e-12, "and dividing beta by s puts it back exactly: {fixed}");

        // The optimum, meanwhile, has not moved at all.
        let exact = crate::exact::Elimination::default();
        assert_eq!(
            exact.ground_state(&base).unwrap().ground_state,
            exact.ground_state(&scaled).unwrap().ground_state,
            "the same state minimises both"
        );
    }

    #[test]
    fn scaling_is_refused_when_it_cannot_land_on_whole_numbers() {
        // The honest failure. An integral fabric and a program whose couplings are not in a fixed
        // ratio cannot both be satisfied by one factor, and a factor that quietly rounded would
        // change the problem rather than move it.
        let mut f = Fabric::unconstrained("integral", Z1_SPICE);
        f.coupling_range = Some(Range::integers(-7.0, 7.0));
        assert_eq!(f.scale_to_fit(&program(&[1.0, 3.7], &[])), None, "3.7 lands nowhere");
        // but a program already on whole numbers scales cleanly
        assert_eq!(f.scale_to_fit(&program(&[2.0, 14.0], &[])), Some(0.5));
    }

    #[test]
    fn a_fabric_with_no_range_declares_none_rather_than_a_guess() {
        // An invented limit is worse than an absent one: it refuses programs that would have run.
        let f = Fabric::unconstrained("sim", Z1_SPICE);
        assert_eq!(f.coupling_range, None);
        assert!(f.check(&program(&[1e9], &[])).is_empty(), "a simulator has no range to violate");
        assert_eq!(f.scale_to_fit(&program(&[1e9], &[])), None, "and nothing to scale toward");
    }

    #[test]
    fn the_declared_fabrics_match_their_published_specifications() {
        // These numbers come from D-Wave's own documentation and are cited at their definitions.
        // The test exists so that changing one is a deliberate act with a diff, not a drift.
        let a2 = Fabric::dwave_advantage2(Z1_SPICE);
        assert_eq!(a2.max_spins, Some(4_400), "Zephyr-12, 4,400+ qubits");
        assert_eq!(a2.max_degree, Some(20), "20-way connectivity");
        assert_eq!(a2.coupling_range, Some(Range::continuous(-1.0, 1.0)), "j_range");
        assert_eq!(a2.field_range, Some(Range::continuous(-4.0, 4.0)), "h_range");
        assert_eq!(a2.coupling_bits, None, "D-Wave publishes no bit count; claiming one invents it");

        let a1 = Fabric::dwave_advantage(Z1_SPICE);
        assert_eq!(a1.max_spins, Some(5_640), "Pegasus, 5,640 qubits");
        assert_eq!(a1.max_degree, Some(15), "15-way connectivity");
        assert!(a1.max_spins > a2.max_spins && a1.max_degree < a2.max_degree,
                "Advantage has more qubits and fewer couplers each; Advantage2 reverses the trade");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::Z1_SPICE;
    use crate::schedule::Schedule;

    fn prog(src: &str) -> Program {
        Program::from_ftp(src).unwrap()
    }

    /// A fabric shaped like a real one: sparse, pairwise, int8 couplings, no field.
    fn constrained() -> Fabric {
        Fabric {
            name: "test-fabric",
            topology: Topology::Degree(4),
            max_spins: Some(64),
            max_degree: Some(4),
            coupling_bits: Some(8),
            field_bits: Some(8),
            supports_field: false,
            max_arity: 2,
            coupling_range: None,
            field_range: None,
            uniform_couplings: false,
            prices: Z1_SPICE,
        }
    }

    #[test]
    fn a_simulator_accepts_anything() {
        let p = prog("ftp 1\nspins 5\nfactor 1 0 1 2 3 4\nbias 0 0.5\n");
        assert!(Fabric::unconstrained("sim", Z1_SPICE).check(&p).is_empty());
    }

    #[test]
    fn every_limit_is_reported_and_names_itself() {
        let mut src = String::from("ftp 1\nspins 100\n");
        for j in 1..=8 {
            src.push_str(&format!("factor 1 0 {j}\n")); // degree 8 on node 0
        }
        src.push_str("factor 1 10 11 12\n"); // arity 3
        src.push_str("bias 5 0.5\n"); // a field
        let bad = constrained().check(&prog(&src));

        assert!(bad.iter().any(|u| matches!(u, Unsupported::TooManySpins { .. })));
        assert!(bad.iter().any(|u| matches!(u, Unsupported::TooHighDegree { .. })));
        assert!(bad.iter().any(|u| matches!(u, Unsupported::ArityTooHigh { .. })));
        assert!(bad.iter().any(|u| matches!(u, Unsupported::NoFieldSupport { .. })));
        assert_eq!(bad.len(), 4, "every violation at once, not just the first: {bad:?}");

        // and each says what to do about it
        let text = bad.iter().map(|u| u.to_string()).collect::<Vec<_>>().join(" | ");
        assert!(text.contains("sparsify"), "the degree error should suggest a fix: {text}");
        assert!(text.contains("pairwise"), "the arity error should suggest a fix: {text}");
    }

    #[test]
    fn int8_precision_is_caught_before_it_changes_the_answer() {
        // The QBoson case: couplings spanning a wide dynamic range cannot survive 8 bits, and
        // nothing about running the model would tell you.
        let p = prog("ftp 1\nspins 3\nfactor 1000 0 1\nfactor 0.5 1 2\n");
        let bad = constrained().check(&p);
        let prec = bad.iter().find(|u| matches!(u, Unsupported::CouplingPrecision { .. }));
        assert!(prec.is_some(), "a 2000:1 range in 8 bits must be refused: {bad:?}");
        assert!(prec.unwrap().to_string().contains("requantize"));
    }

    #[test]
    fn a_narrow_range_survives_int8_and_is_not_refused() {
        // The check must not fire on models that are fine, or callers will learn to ignore it.
        let p = prog("ftp 1\nspins 4\nfactor 1 0 1\nfactor -1 1 2\nfactor 1 2 3\n");
        assert!(!constrained()
            .check(&p)
            .iter()
            .any(|u| matches!(u, Unsupported::CouplingPrecision { .. })));
    }

    #[test]
    fn requantizing_reports_the_damage_it_did() {
        let mut p = prog("ftp 1\nspins 3\nfactor 1000 0 1\nfactor 0.5 1 2\n");
        let before: Vec<f64> = p.factors.iter().map(|f| f.weight()).collect();
        let err = constrained().requantize(&mut p);
        let after: Vec<f64> = p.factors.iter().map(|f| f.weight()).collect();
        assert!(err > 1e-3, "it should admit a real loss, got {err}");
        assert_ne!(before, after, "and it should actually have changed the weights");
        // afterwards the program fits the fabric it was quantised for
        assert!(!constrained()
            .check(&p)
            .iter()
            .any(|u| matches!(u, Unsupported::CouplingPrecision { .. })));
    }

    #[test]
    fn the_cpu_backend_runs_a_program_through_the_trait() {
        let mut d = Cpu::default();
        let p = prog("ftp 1\nspins 5\nfactor -1 0 1\nfactor -1 1 2\nfactor -1 2 3\n\
                      factor -1 3 4\nfactor -1 4 0\n");
        assert!(d.program(&p).is_empty());
        let s = d.run(&Schedule::geometric(0.05, 6.0, 60, 40), 1).unwrap();
        assert_eq!(s.len(), 5);
        let g = p.to_graph().unwrap();
        assert_eq!(g.energy(&s), -3.0, "the frustrated 5-cycle optimum, through the Device seam");
        assert!(d.ledger().samples > 0, "the ledger must be charged");
    }

    #[test]
    fn a_backend_that_cannot_run_it_says_so_before_running() {
        let mut d = Cpu::default();
        // arity 3 cannot become a graph; the refusal must come from `program`, not from `run`
        let p = prog("ftp 1\nspins 4\nfactor 1 0 1 2\n");
        let bad = d.program(&p);
        assert!(!bad.is_empty(), "a program it cannot lower must be refused up front");
        assert!(d.run(&Schedule::constant(1.0, 10), 1).is_err());
    }
}
