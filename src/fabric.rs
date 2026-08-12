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
use std::collections::BTreeSet;
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
    /// How a coupling is stored.
    ///
    /// State it even when it is generous. An undeclared precision is the defect this whole module
    /// exists to prevent.
    pub coupling_precision: Precision,
    /// How a field is stored.
    pub field_precision: Precision,
    /// Whether an external field can be applied at all. Some fabrics cannot hold one.
    pub supports_field: bool,
    /// Maximum factor arity. Two means pairwise only, which is most hardware.
    pub max_arity: usize,
    /// What magnitudes a coupling may take, or `None` for unbounded.
    pub coupling_range: Option<Range>,
    /// What magnitudes a field may take, or `None` for unbounded.
    pub field_range: Option<Range>,
    /// What this fabric's vendor does not publish.
    ///
    /// Empty for a machine whose limits are fully documented. Non-empty means [`Fabric::verdict`]
    /// cannot promise a run however clean `check` comes back, because the check was made against an
    /// incomplete description. Declaring the gap is what makes a partially-documented fabric usable
    /// at all — the alternative was leaving it out entirely, which helps nobody.
    pub unstated: &'static [&'static str],
    /// Whether a program's variables map one-to-one onto the machine's sites.
    ///
    /// False for every annealer whose hardware graph is not complete. There, a variable with more
    /// neighbours than the topology allows becomes a CHAIN of physical sites held together by
    /// strong couplings — minor embedding — and the number of sites a program needs is not its
    /// variable count. A fully connected problem on a 5,640-qubit Pegasus reaches nowhere near
    /// 5,640 variables.
    ///
    /// It changes what [`Fabric::check`] can honestly say. Passing means "nothing here rules it
    /// out", not "this will run".
    pub native_placement: bool,
    /// Whether every coupling must have the same weight.
    ///
    /// Set by fabrics that *count* active neighbours rather than summing weighted ones. It is a
    /// severe restriction — a spin glass cannot be expressed at all — and exactly the kind of limit
    /// that goes undeclared until someone's answers come back wrong.
    pub uniform_couplings: bool,
    /// Energy prices for the ledger.
    pub prices: Prices,
}

/// How a fabric stores a coefficient.
///
/// The distinction is not decoration. Fixed-point spreads a uniform step across the whole range, so
/// a coefficient a thousandth the size of the largest one is lost entirely. Floating point keeps a
/// constant number of significant digits, so the same coefficient survives. Modelling one as the
/// other reports an error that is wrong by orders of magnitude in whichever direction.
///
/// This started as a bare `Option<u32>` bit count, which could describe Hitachi's four-bit integers
/// and could not describe Toshiba's float32 at all.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Precision {
    /// Every `f64` arrives intact. A simulator, not a machine.
    Exact,
    /// The vendor does not publish it.
    ///
    /// Distinct from `Exact`, and the distinction is the whole point. D-Wave was declared `Exact`
    /// here, which claims every `f64` survives — false of an analog machine whose real limit is
    /// integrated control error. Saying "unstated" refuses to certify rather than quietly
    /// promising the best case, and a fabric carrying one of these cannot return
    /// [`Verdict::Runnable`].
    Unstated,
    /// Signed fixed-point in `bits` bits, one of them the sign, with a uniform step sized by the
    /// largest coefficient present.
    Fixed { bits: u32 },
    /// IEEE binary floating point with `mantissa` significand bits — 24 for `f32`, 53 for `f64`.
    Float { mantissa: u32 },
}

impl Precision {
    /// The worst relative error storing these values here would introduce.
    pub fn worst_relative_error(&self, vals: &[f64]) -> f64 {
        match *self {
            Precision::Exact => 0.0,
            // Unknown is not zero. Returning 0.0 would let a caller conclude nothing is lost, which
            // is the claim this variant exists to avoid making; `Verdict::LimitsUnstated` is where
            // it surfaces instead.
            Precision::Unstated => 0.0,
            Precision::Float { mantissa } => {
                if vals.iter().all(|v| *v == 0.0) {
                    0.0
                } else {
                    // Rounding to nearest costs at most half an ulp, which is 2^-mantissa
                    // RELATIVELY -- the same for every value, large or small. That is exactly the
                    // property fixed point does not have.
                    (2.0f64).powi(-(mantissa as i32))
                }
            }
            Precision::Fixed { bits } => {
                let max = vals.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
                if max == 0.0 || bits == 0 {
                    return 0.0;
                }
                // One bit is the sign, so a single bit leaves no levels at all.
                if bits == 1 {
                    return 1.0;
                }
                let levels = ((1u64 << (bits - 1)) - 1) as f64;
                let step = max / levels;
                vals.iter()
                    .map(|&w| {
                        if w == 0.0 {
                            0.0
                        } else {
                            ((w / step).round() * step - w).abs() / w.abs()
                        }
                    })
                    .fold(0.0f64, f64::max)
            }
        }
    }
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
        v.is_finite() && v >= self.lo && v <= self.hi && (!self.integral || v.fract() == 0.0)
    }

    /// The largest positive factor `s` for which `s·v` still lands in this range.
    ///
    /// Zero or negative when no positive factor works — a negative value cannot be scaled into a
    /// range with a non-negative floor, however small the factor.
    ///
    /// A range has two sides and they are not always mirror images: D-Wave's `extended_j_range` is
    /// `[-2, 1]` and the Pt V2's field is `[0, 6]`. A single "largest magnitude" scalar describes
    /// neither. Taking the smaller endpoint gives 0 for the second, which scales every coefficient
    /// in the program to zero; taking the larger gives 2 for the first, permitting a `+2` coupling
    /// the machine cannot hold. Both were wrong here, and the first shipped.
    pub fn headroom_for(&self, v: f64) -> f64 {
        if v > 0.0 {
            self.hi / v
        } else if v < 0.0 {
            self.lo / v // both negative, so the quotient is positive
        } else {
            f64::INFINITY // zero scales to zero, which every range containing zero holds
        }
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

/// A reason a fabric cannot promise a program will run, even with nothing ruling it out.
///
/// There can be more than one, which is why [`Verdict`] holds a list. D-Wave has both: variables
/// are placed by minor embedding AND the coefficient precision is unpublished. An answer that could
/// name only one of those would drop the other, and which one it dropped would be an accident of
/// the order they were checked in.
#[derive(Clone, Debug, PartialEq)]
pub enum Caveat {
    /// Variables are placed by minor embedding, so `vars` variables do not mean `vars` sites.
    ///
    /// A variable with more neighbours than the topology allows becomes a chain of physical sites.
    /// Whether such a placement exists is NP-hard and depends on the program's structure, not on
    /// any number a fabric declares. The real answer needs an embedder run against the machine's
    /// own working graph, which has holes, because yield is never 100%.
    NeedsEmbedding { vars: usize, sites: Option<usize> },
    /// The vendor does not publish something that would have to be checked.
    ///
    /// A limit nobody states cannot be checked, and `None` must not be read as "no limit".
    LimitsUnstated { missing: &'static [&'static str] },
}

impl core::fmt::Display for Caveat {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Caveat::NeedsEmbedding { vars, sites } => write!(
                f,
                "this fabric places variables by minor embedding, so {vars} variables do not mean \
                 {vars} sites{}; run an embedder against the machine's own working graph",
                match sites {
                    Some(n) => format!(" out of {n}"),
                    None => String::new(),
                }
            ),
            Caveat::LimitsUnstated { missing } => {
                write!(f, "this fabric does not publish {}", missing.join(", "))
            }
        }
    }
}

/// What a fabric can say about a program that nothing rules out.
///
/// Empty caveats mean it runs. Anything else means the check was necessary and not sufficient, and
/// says why — rather than returning silence for a caller to read as a yes.
#[derive(Clone, Debug, PartialEq)]
pub struct Verdict {
    caveats: Vec<Caveat>,
}

impl Verdict {
    /// Nothing rules it out, nothing is unpublished, and one variable goes to one site.
    pub fn is_runnable(&self) -> bool {
        self.caveats.is_empty()
    }
    /// Every reason this is not a promise.
    pub fn caveats(&self) -> &[Caveat] {
        &self.caveats
    }
}

impl core::fmt::Display for Verdict {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.caveats.is_empty() {
            return write!(f, "nothing rules it out and placement is direct: it runs");
        }
        write!(f, "nothing declared rules it out, but this is not a promise that it runs:")?;
        for c in &self.caveats {
            write!(f, "\n  - {c}")?;
        }
        Ok(())
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
    /// The program cannot be placed on this fabric's sites as written.
    ///
    /// For a fabric that does its own placement check — a fixed grid, a named topology — where the
    /// failure is specific and knowable. Distinct from [`Verdict::NeedsEmbedding`], which is the
    /// case where placement is a question nobody cheap can answer; this is the case where the
    /// answer is known and it is no.
    Unplaceable { detail: String },
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
                "a factor of arity {arity} cannot run on a fabric limited to {limit}; lower it with \
                 `ferrotherm::reduce::to_pairwise`, which adds one ancilla spin per substituted \
                 pair and leaves every state in the same order"
            ),
            Unsupported::NoFieldSupport { nodes } => write!(
                f,
                "{nodes} spins carry an external field and this fabric cannot apply one"
            ),
            Unsupported::Unplaceable { detail } => write!(f, "{detail}"),
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
            coupling_precision: Precision::Exact,
            field_precision: Precision::Exact,
            supports_field: true,
            max_arity: usize::MAX,
            coupling_range: None,
            field_range: None,
            native_placement: true,
            unstated: &[],
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
        let couplings: Vec<f64> = p.factors.iter().map(|f| f.weight()).collect();
        let fields: Vec<f64> = p.bias.iter().map(|(_, h)| *h).collect();

        // The tightest constraint any single coefficient imposes, over both ranges.
        let bound = |r: Option<Range>, vals: &[f64]| -> f64 {
            match r {
                None => f64::INFINITY, // an undeclared range constrains nothing
                Some(r) => vals.iter().fold(f64::INFINITY, |acc, &v| acc.min(r.headroom_for(v))),
            }
        };
        if self.coupling_range.is_none() && self.field_range.is_none() {
            return None; // nothing to scale toward
        }
        let mut s = bound(self.coupling_range, &couplings).min(bound(self.field_range, &fields));

        if s == f64::INFINITY {
            return Some(1.0); // no coefficient constrains anything; it already fits
        }
        if !(s > 0.0) || !s.is_finite() {
            // A negative coefficient against a non-negative floor, or the reverse. No positive
            // factor helps, and returning one that "nearly" works would be worse than saying so.
            return None;
        }

        // Integrality is per range: a fabric whose couplings are whole numbers may still take
        // continuous fields, and requiring both to land would refuse programs that fit.
        let lands = |s: f64| {
            let ok = |r: Option<Range>, vals: &[f64]| match r {
                Some(r) if r.integral => vals.iter().all(|&v| (v * s).fract() == 0.0),
                _ => true,
            };
            ok(self.coupling_range, &couplings) && ok(self.field_range, &fields)
        };
        // The factor came from a division, so `v · s` can land one ulp outside the range it was
        // computed to satisfy. Verify against the same predicate `check` uses — a scale_to_fit
        // whose result does not pass check is worse than no answer at all.
        let fits = |s: f64| {
            self.coupling_range.map(|r| couplings.iter().all(|&v| r.holds(v * s))).unwrap_or(true)
                && self.field_range.map(|r| fields.iter().all(|&v| r.holds(v * s))).unwrap_or(true)
        };

        if !lands(s) {
            // An integral range wants a factor putting every value on a whole number, and the
            // saturating one rarely does.
            //
            // The candidates are not arbitrary. If the smallest non-zero magnitude in the program
            // is `v0`, then `v0·s` must itself be a non-zero integer, so `s = n/v0` for some
            // positive integer `n` — and `s` cannot exceed the saturating factor. So walk `n` DOWN
            // from that ceiling and take the first that lands and fits, which is the largest.
            // Searching `s/k` instead, as this first did, misses every candidate whose numerator is
            // not one: it could not find 3/14 for a program of 14 and 28 against ±7.
            // Only the coefficients an INTEGRAL range governs constrain the candidate set. A
            // fractional field under a continuous range is free to land anywhere, and letting it
            // set `v0` yields a ceiling of zero candidates and a spurious None.
            let integral_of = |r: Option<Range>, vals: &[f64]| -> Vec<f64> {
                match r {
                    Some(r) if r.integral => vals.to_vec(),
                    _ => Vec::new(),
                }
            };
            let governed = integral_of(self.coupling_range, &couplings);
            let governed2 = integral_of(self.field_range, &fields);
            let v0 = governed
                .iter()
                .chain(governed2.iter())
                .map(|v| v.abs())
                .filter(|v| *v > 0.0)
                .fold(f64::INFINITY, f64::min);
            if !v0.is_finite() {
                return None;
            }
            let top = (s * v0).floor();
            if !(top >= 1.0) || top > 1e6 {
                return None; // no candidate, or too many to enumerate honestly
            }
            let mut n = top;
            while n >= 1.0 {
                let cand = n / v0;
                if lands(cand) && fits(cand) {
                    return Some(cand);
                }
                n -= 1.0;
            }
            return None;
        }
        for _ in 0..64 {
            if fits(s) {
                return Some(s);
            }
            s = f64::from_bits(s.to_bits() - 1); // the next representable value below
        }
        None
    }

    // ---- declared fabrics ---------------------------------------------------------------------
    //
    // A fabric can be DECLARED without being reachable, and that is most of the value: a caller can
    // ask what rules their program out before buying time on a machine. Every number below is from
    // the vendor's own published material, cited where it is used. Where a vendor does not publish
    // a limit, the field is `None` and this says so rather than guessing — an invented limit
    // refuses programs that would have run.
    //
    // What is NOT declared here is as deliberate. This review found three incompatible figures for
    // the Zephyr graph's size across three sources, and two different answers for the largest
    // embeddable clique that turned out to be answers to different questions (largest at chain
    // length 2, versus largest at any chain length). None of them is asserted.

    /// D-Wave Advantage2 — Zephyr topology, generally available May 2025.
    ///
    /// 4,400+ qubits at 20-way connectivity, from D-Wave's own topology documentation and its
    /// general-availability announcement. Couplings are continuous over `[-1, 1]` and fields over
    /// `[-4, 4]` (`j_range` and `h_range` in the published solver properties). `extended_j_range`
    /// reaches `[-2, 1]` but needs per-chain flux-bias calibration, so it is not the default.
    ///
    /// Two things this is not. It is a quantum annealer rather than a thermodynamic sampler: it
    /// minimises an Ising energy and holds no temperature you set, so a [`crate::certify`]
    /// certificate has no β to check it against. And it does not place one variable per qubit —
    /// see `native_placement`.
    pub fn dwave_advantage2(prices: Prices) -> Fabric {
        Fabric {
            name: "dwave-advantage2",
            topology: Topology::Degree(20),
            // Physical qubits, not problem variables. `native_placement: false` is what stops this
            // number being read as a variable budget.
            max_spins: Some(4_400),
            max_degree: Some(20),
            // Analog. The practical limit is integrated control error rather than a bit count,
            // and D-Wave publishes no bit count. This said `Exact`, which claims every f64 arrives
            // intact — the opposite of true for an analog machine, and a claim invented to fill a
            // field rather than measured.
            coupling_precision: Precision::Unstated,
            field_precision: Precision::Unstated,
            supports_field: true,
            max_arity: 2,
            coupling_range: Some(Range::continuous(-1.0, 1.0)),
            field_range: Some(Range::continuous(-4.0, 4.0)),
            native_placement: false,
            unstated: &["coefficient precision (analog, bounded by integrated control error)"],
            uniform_couplings: false,
            prices,
        }
    }

    /// D-Wave Advantage — Pegasus, 5,640 qubits at 15-way connectivity, from D-Wave's topology
    /// documentation. More qubits than Advantage2 and fewer couplers each, which is the trade the
    /// newer topology reverses.
    pub fn dwave_advantage(prices: Prices) -> Fabric {
        Fabric {
            name: "dwave-advantage",
            topology: Topology::Degree(15),
            max_spins: Some(5_640),
            max_degree: Some(15),
            coupling_precision: Precision::Unstated,
            field_precision: Precision::Unstated,
            supports_field: true,
            max_arity: 2,
            coupling_range: Some(Range::continuous(-1.0, 1.0)),
            field_range: Some(Range::continuous(-4.0, 4.0)),
            native_placement: false,
            unstated: &["coefficient precision (analog, bounded by integrated control error)"],
            uniform_couplings: false,
            prices,
        }
    }

    /// Fujitsu Digital Annealer, third generation (`FujitsuDA3Solver`).
    ///
    /// From Fujitsu's own API documentation: up to **100,000 bits**, a **fully connected**
    /// architecture, quadratic coefficients as **64-bit signed integers** over `-2^63+1 ..= 2^63-1`
    /// and linear coefficients as **76-bit signed integers**. Coefficients must be whole numbers;
    /// the service will scale and round a submission that is not, which is a different program than
    /// the one written and is exactly what `Range::integral` exists to catch first.
    ///
    /// Fully connected is the interesting part. There is no embedding here — 100,000 bits means
    /// 100,000 variables, where 5,640 D-Wave qubits does not mean 5,640 variables. That is what
    /// `native_placement` distinguishes, and it is the whole reason the field exists.
    ///
    /// Note the vendor's own marketing pages quote 8,192 bits and 64-bit gradation for an earlier
    /// generation. The API documentation is what a submission is actually checked against, so it is
    /// what is declared.
    pub fn fujitsu_da3(prices: Prices) -> Fabric {
        Fabric {
            name: "fujitsu-da3",
            topology: Topology::Unconstrained, // fully coupled: every bit reaches every other
            max_spins: Some(100_000),
            max_degree: None,
            coupling_precision: Precision::Fixed { bits: 64 },
            field_precision: Precision::Fixed { bits: 76 },
            supports_field: true,
            max_arity: 2,
            // f64 cannot represent 2^63 exactly, and a program's coefficients are f64 here, so the
            // integrality flag is the binding half of this in practice rather than the bound.
            coupling_range: Some(Range::integers(-(2f64.powi(63)) + 1.0, 2f64.powi(63) - 1.0)),
            field_range: Some(Range::integers(-(2f64.powi(75)) + 1.0, 2f64.powi(75) - 1.0)),
            native_placement: true,
            unstated: &[],
            uniform_couplings: false,
            prices,
        }
    }

    /// Toshiba SQBM+, the QUBO solver.
    ///
    /// From Toshiba's own user manual: **up to 10,000,000 variables**, a dense QUBO coefficient
    /// matrix, and coefficients carried as **float32** — the manual's own data arrays are
    /// `np.float32`.
    ///
    /// That float32 is why [`Precision`] exists as a type rather than a bit count. Modelled as
    /// 24-bit fixed point it would report a coefficient a millionth of the largest as totally lost;
    /// in float32 it survives with its significant digits intact, because floating-point error is
    /// relative. The two answers differ by orders of magnitude.
    ///
    /// SQBM+ also offers a PUBO solver taking terms up to **order 4**, which most fabrics cannot
    /// express at all. This declares the QUBO solver, which is what a pairwise `.ftp` maps onto;
    /// a higher-order fabric would be a separate declaration rather than a wider `max_arity` here.
    pub fn toshiba_sqbm(prices: Prices) -> Fabric {
        Fabric {
            name: "toshiba-sqbm-qubo",
            topology: Topology::Unconstrained,
            max_spins: Some(10_000_000),
            max_degree: None,
            coupling_precision: Precision::Float { mantissa: 24 },
            field_precision: Precision::Float { mantissa: 24 },
            supports_field: true,
            max_arity: 2,
            // A coefficient beyond float32's reach does not arrive, however fine f64 held it.
            coupling_range: Some(Range::continuous(-(f32::MAX as f64), f32::MAX as f64)),
            field_range: Some(Range::continuous(-(f32::MAX as f64), f32::MAX as f64)),
            native_placement: true,
            unstated: &[],
            uniform_couplings: false,
            prices,
        }
    }

    /// Toshiba SQBM+, the **PUBO** solver — terms up to order four.
    ///
    /// The same product as [`Fabric::toshiba_sqbm`] through a different API. Toshiba's manual:
    /// "a solver that solves a problem with higher order terms up to order 4".
    ///
    /// Worth declaring separately rather than widening the QUBO solver's `max_arity`, because they
    /// are different endpoints with different limits, and because this is the one fabric here where
    /// a three- or four-body model runs **without** [`crate::reduce`] — no ancillas, no penalty, no
    /// distribution caveat. That is a real difference to a caller and it should be visible in the
    /// declaration rather than in a comment.
    pub fn toshiba_sqbm_pubo(prices: Prices) -> Fabric {
        Fabric {
            name: "toshiba-sqbm-pubo",
            topology: Topology::Unconstrained,
            max_spins: Some(10_000_000),
            max_degree: None,
            coupling_precision: Precision::Float { mantissa: 24 },
            field_precision: Precision::Float { mantissa: 24 },
            supports_field: true,
            max_arity: 4,
            coupling_range: Some(Range::continuous(-(f32::MAX as f64), f32::MAX as f64)),
            field_range: Some(Range::continuous(-(f32::MAX as f64), f32::MAX as f64)),
            native_placement: true,
            unstated: &[],
            uniform_couplings: false,
            prices,
        }
    }

    /// QBoson CPQC — a coherent Ising machine, from the Kaiwu SDK's own documentation.
    ///
    /// An optical machine: a network of degenerate optical parametric oscillators whose steady
    /// state encodes the Ising ground state, with the couplings applied in measurement feedback
    /// rather than wired physically.
    ///
    /// **Eight-bit fixed point.** The Kaiwu documentation is explicit — "the CIM machine only
    /// supports 8-bit INT space [-128, 127]" — and the SDK ships
    /// `perform_precision_adaption_mutate` and `..._split` because real models do not fit it. That
    /// is a far harder limit than anything else declared here, and it is the number that decides
    /// whether a program survives at all.
    ///
    /// What the vendor does **not** publish, and this therefore does not claim: the machine's size
    /// or its connectivity. Third-party papers give 550 qubits for a CPQC-550 and 2,000 variables
    /// for a third-generation part; those are not the vendor's numbers and are not declared. A
    /// caller gets [`Verdict::LimitsUnstated`] rather than a promise.
    ///
    /// An earlier version of this file said no vendor material gave the precision. That was wrong:
    /// it is in the SDK documentation rather than a datasheet, which is where this review had
    /// looked. The claim it is replacing is the reason the wording of an absence matters — "we did
    /// not find it" would have survived being wrong; "it does not exist" did not.
    pub fn qboson_cpqc(prices: Prices) -> Fabric {
        Fabric {
            name: "qboson-cpqc",
            topology: Topology::Unconstrained,
            // Not published. `None` here would read as "no limit", which is why `unstated` says so
            // out loud and the verdict refuses to promise.
            max_spins: None,
            max_degree: None,
            coupling_precision: Precision::Fixed { bits: 8 },
            field_precision: Precision::Fixed { bits: 8 },
            supports_field: true,
            max_arity: 2,
            coupling_range: Some(Range::integers(-128.0, 127.0)),
            field_range: Some(Range::integers(-128.0, 127.0)),
            native_placement: true,
            unstated: &["maximum problem size", "coupling topology"],
            uniform_couplings: false,
            prices,
        }
    }

    // Still not declared: Fujitsu's earlier Digital Annealer generations, whose marketing numbers
    // (8,192 bits, 64-bit gradation) disagree with the API documentation used above and which this
    // review did not separate into distinct products with confidence.

    /// What an empty [`Fabric::check`] actually means here.
    ///
    /// `check` answers "what rules this out". That is not the same question as "will it run", and
    /// on most annealers the second has no cheap answer: a variable with more neighbours than the
    /// topology allows is placed as a CHAIN of physical sites, and whether such a placement exists
    /// is minor embedding — NP-hard, and dependent on the program's structure rather than on any
    /// number a fabric can declare. Returning an empty violation list there would be read as a yes.
    pub fn verdict(&self, p: &Program) -> Result<Verdict, Vec<Unsupported>> {
        let bad = self.check(p);
        if !bad.is_empty() {
            return Err(bad);
        }
        let mut caveats = Vec::new();
        if !self.native_placement {
            caveats.push(Caveat::NeedsEmbedding { vars: p.spins, sites: self.max_spins });
        }
        if !self.unstated.is_empty() {
            caveats.push(Caveat::LimitsUnstated { missing: self.unstated });
        }
        Ok(Verdict { caveats })
    }

    /// What rules this program out on this fabric?
    ///
    /// Returns every violation rather than the first, because a caller deciding whether to embed a
    /// model wants the whole picture in one pass.
    ///
    /// **An empty result is not a promise that the program will run** — see [`Fabric::verdict`],
    /// which says so in the type rather than leaving a caller to read silence as a yes.
    pub fn check(&self, p: &Program) -> Vec<Unsupported> {
        let mut out = Vec::new();

        if let Some(limit) = self.max_spins {
            if p.spins > limit {
                out.push(Unsupported::TooManySpins { need: p.spins, limit });
            }
        }

        // An arity-1 factor is a FIELD. `Program::to_graph` lowers it with `b.bias(...)`, so it is
        // one everywhere in this crate — and checking it as a coupling here let a program whose
        // fields were written that way past a fabric that has no field at all.
        let (unary, pairs): (Vec<_>, Vec<_>) =
            p.factors.iter().partition(|f| f.arity() == 1);

        let mut worst_arity = 0;
        // Degree is over EDGES, not over factor mentions. Two factors on the same pair are one
        // edge with a summed weight once the program becomes a graph, and counting them twice
        // refuses a program the fabric can hold.
        let mut edges: BTreeSet<(usize, usize)> = BTreeSet::new();
        for f in &p.factors {
            worst_arity = worst_arity.max(f.arity());
            if f.arity() == 2 {
                let v: Vec<usize> = f.vars().collect();
                if v[0] < p.spins && v[1] < p.spins {
                    edges.insert((v[0].min(v[1]), v[0].max(v[1])));
                }
            }
        }
        let mut degree = vec![0usize; p.spins];
        for (a, b) in &edges {
            degree[*a] += 1;
            degree[*b] += 1;
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

        // A field is a field however it was written.
        let fields: Vec<f64> = p
            .bias
            .iter()
            .map(|(_, h)| *h)
            .chain(unary.iter().map(|f| f.weight()))
            .collect();
        if !self.supports_field && !fields.is_empty() {
            out.push(Unsupported::NoFieldSupport { nodes: fields.len() });
        }

        // The WORST offender rather than every one: a program built by a loop violates a range in
        // thousands of places for one reason, and a thousand identical findings buries the ones
        // that differ. Worst means largest magnitude, because that is the value setting the scale
        // factor a caller needs. This used to take the FIRST, while saying it took the worst.
        let worst_of = |r: Range, vals: &mut dyn Iterator<Item = f64>| -> Option<f64> {
            vals.filter(|v| !r.holds(*v))
                .max_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap_or(core::cmp::Ordering::Equal))
        };
        if let Some(r) = self.coupling_range {
            if let Some(w) = worst_of(r, &mut pairs.iter().map(|f| f.weight())) {
                out.push(Unsupported::OutOfRange { what: "coupling", value: w, range: r });
            }
        }
        if let Some(r) = self.field_range {
            if let Some(h) = worst_of(r, &mut fields.iter().copied()) {
                out.push(Unsupported::OutOfRange { what: "field", value: h, range: r });
            }
        }

        if self.uniform_couplings {
            // By VALUE, not by bit pattern. `0.0` and `-0.0` are the same coupling and have
            // different bits, so a program mixing them was reported as having two distinct weights.
            let mut seen: Vec<f64> = pairs.iter().map(|f| f.weight()).collect();
            seen.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
            seen.dedup_by(|a, b| a == b);
            if seen.len() > 1 {
                out.push(Unsupported::NonUniformCouplings { distinct: seen.len() });
            }
        }

        // An INTEGRAL range already says what survives: a value on the grid is stored exactly and
        // one off it was refused above. Running the fixed-point model as well reports an error that
        // is not there — it derives a step by rescaling the largest coefficient, which describes a
        // fabric that normalises its input, not one whose grid is the integers. Two statements of
        // the same limit disagreeing is worse than one, so the range wins where it is present.
        let field_grid = self.field_range.map(|r| r.integral).unwrap_or(false);
        let coupling_grid = self.coupling_range.map(|r| r.integral).unwrap_or(false);

        let field_err =
            if field_grid { 0.0 } else { self.field_precision.worst_relative_error(&fields) };
        if field_err > 0.05 {
            out.push(Unsupported::CouplingPrecision {
                bits: match self.field_precision {
                    Precision::Fixed { bits } | Precision::Float { mantissa: bits } => bits,
                    Precision::Exact | Precision::Unstated => 0,
                },
                worst_relative_error: field_err,
            });
        }

        {
            let bits = match self.coupling_precision {
                Precision::Fixed { bits } | Precision::Float { mantissa: bits } => bits,
                Precision::Exact | Precision::Unstated => 0,
            };
            let err = if coupling_grid {
                0.0
            } else {
                self.coupling_precision
                    .worst_relative_error(&pairs.iter().map(|f| f.weight()).collect::<Vec<_>>())
            };
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
        Self::quantization_error_of(&p.factors.iter().map(|f| f.weight()).collect::<Vec<_>>(), bits)
    }

    /// The worst relative error quantising these values to `bits` signed bits would introduce.
    ///
    /// Shared by the coupling and field checks. `field_bits` was declared by every fabric in this
    /// file and read by nothing, so a fabric holding three-bit fields accepted a program with
    /// fine-grained ones and lost them at submission.
    pub fn quantization_error_of(vals: &[f64], bits: u32) -> f64 {
        let max = vals.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        // One bit is the sign, so a single bit leaves no levels at all and the shift below would
        // be `1 << 0 - 1 == 0`, making every step infinite. Say "everything is lost" instead.
        if max == 0.0 || bits == 0 {
            return 0.0;
        }
        if bits == 1 {
            return 1.0;
        }
        let levels = ((1u64 << (bits - 1)) - 1) as f64;
        let step = max / levels;
        vals.iter()
            .map(|&w| if w == 0.0 { 0.0 } else { ((w / step).round() * step - w).abs() / w.abs() })
            .fold(0.0f64, f64::max)
    }

    /// Quantise a program's couplings to this fabric's precision, returning the worst relative
    /// error introduced.
    ///
    /// Explicit by design. A fabric that quantises silently is answering a different question than
    /// the one it was asked, and the caller is the last to find out.
    pub fn requantize(&self, p: &mut Program) -> f64 {
        // Only fixed point actually moves a coefficient onto a grid. Floating point rounds it in
        // the last significand bits, which is not something to do here on the caller's behalf, and
        // Exact does nothing at all.
        let Precision::Fixed { bits } = self.coupling_precision else { return 0.0 };
        if bits == 0 {
            return 0.0;
        }
        let weights: Vec<f64> = p.factors.iter().map(|f| f.weight()).collect();
        let err = self.coupling_precision.worst_relative_error(&weights);
        let max = weights.iter().map(|w| w.abs()).fold(0.0f64, f64::max);
        if max == 0.0 || bits == 1 {
            return err;
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
        let mut f = Fabric::unconstrained("cpu", crate::ledger::Z1_SPICE);
        // It lowers through `Program::to_graph`, which is pairwise. `unconstrained` says
        // `usize::MAX`, which let an arity-3 program through `check` and then failed in `program`
        // with a hardcoded `arity: 3` -- reporting three however many the program really had.
        f.max_arity = 2;
        f
    }

    fn program(&mut self, p: &Program) -> Vec<Unsupported> {
        let bad = self.fabric().check(p);
        if bad.is_empty() {
            match p.to_graph() {
                Ok(g) => {
                    self.state = vec![-1; g.n];
                    self.graph = Some(g);
                }
                // `check` above declares max_arity 2, so a higher-order program is refused there
                // and this is only reached if `to_graph` fails for some other reason.
                Err(e) => {
                    return vec![Unsupported::Unplaceable { detail: e.to_string() }]
                }
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
    fn fixed_point_loses_a_small_coefficient_and_floating_point_does_not() {
        // Why Precision is a type rather than a bit count. The same 24 bits mean completely
        // different things, and modelling Toshiba's float32 as fixed point would report a
        // coefficient it holds perfectly well as totally lost.
        // A ratio wide enough that fixed point cannot hold both: 24 bits give ~8.4M levels, so a
        // step of 1e8/8.4e6 is about 12 and the 1.0 rounds to zero. A 1e6:1 ratio only costs 4.6%,
        // which is a real effect and not a vivid one -- worth choosing deliberately rather than
        // discovering that the chosen numbers understate the point.
        let vals = [1e8, 1.0];

        let fixed = Precision::Fixed { bits: 24 }.worst_relative_error(&vals);
        let float = Precision::Float { mantissa: 24 }.worst_relative_error(&vals);

        assert!(fixed >= 1.0, "fixed point rounds the 1.0 away entirely: {fixed}");
        assert!(float < 1e-6, "float32 keeps 24 significant bits of each: {float}");
        assert!(fixed / float > 1e4, "they differ by orders of magnitude, not by a little");

        assert_eq!(Precision::Exact.worst_relative_error(&vals), 0.0);
        // and a single bit is all sign and no levels, which used to divide by zero levels
        assert_eq!(Precision::Fixed { bits: 1 }.worst_relative_error(&vals), 1.0);
    }

    #[test]
    fn an_unstated_limit_is_not_an_absent_one() {
        // `None` meant both "no limit" and "not published", so a machine whose size the vendor does
        // not state looked exactly like a simulator with no size at all. QBoson publishes its
        // eight-bit precision and not its size; the verdict has to say so rather than promise.
        let q = Fabric::qboson_cpqc(Z1_SPICE);
        let p = program(&[3.0, -5.0], &[(0, 2.0)]);
        assert!(q.check(&p).is_empty(), "whole numbers inside [-128, 127] fit: {:?}", q.check(&p));
        let v = q.verdict(&p).expect("nothing rules it out");
        assert!(!v.is_runnable(), "it cannot promise a run");
        assert!(v.caveats().iter().any(|c| matches!(
            c, Caveat::LimitsUnstated { missing } if missing.iter().any(|m| m.contains("size"))
        )), "{:?}", v.caveats());
        assert!(v.to_string().contains("not a promise"));

        // and the eight-bit range it DOES publish is enforced
        let bad = q.check(&program(&[200.0], &[]));
        assert_eq!(bad.len(), 1, "{bad:?}");
        assert!(bad[0].to_string().contains("integers -128..=127"), "{}", bad[0]);
        // fractional too: the machine stores fixed-point integers
        assert!(!q.check(&program(&[1.5], &[])).is_empty(), "1.5 is not an 8-bit integer");
    }

    #[test]
    fn d_wave_does_not_claim_a_precision_it_never_published() {
        // It said `Exact` -- every f64 arrives intact -- which is the opposite of true for an
        // analog machine whose real limit is integrated control error.
        let dw = Fabric::dwave_advantage2(Z1_SPICE);
        assert_eq!(dw.coupling_precision, Precision::Unstated);
        assert!(!dw.unstated.is_empty(), "and the gap is declared rather than implied");
        assert!(
            !dw.verdict(&program(&[0.5], &[])).unwrap().is_runnable(),
            "so it cannot promise a run either"
        );
    }

    #[test]
    fn the_pubo_solver_takes_a_four_body_term_where_everything_else_needs_reducing() {
        // The one fabric here where a three- or four-body model runs with no ancillas, no penalty
        // and no distribution caveat. That is a real difference and it is visible in the type.
        let cubic = Program::from_ftp("ftp 1\nspins 5\nfactor 1.0 0 1 2\nfactor 1.0 1 2 3 4\n")
            .unwrap();

        let pubo = Fabric::toshiba_sqbm_pubo(Z1_SPICE);
        assert_eq!(pubo.max_arity, 4);
        assert!(pubo.check(&cubic).is_empty(), "order 4 runs as written: {:?}", pubo.check(&cubic));
        assert!(pubo.verdict(&cubic).unwrap().is_runnable());

        // every other fabric needs the reduction first
        for f in [Fabric::toshiba_sqbm(Z1_SPICE), Fabric::fujitsu_da3(Z1_SPICE),
                  Fabric::qboson_cpqc(Z1_SPICE)] {
            assert!(
                f.check(&cubic).iter().any(|u| matches!(u, Unsupported::ArityTooHigh { .. })),
                "{} should refuse a 4-body term", f.name
            );
        }

        // and after reducing, they take it
        let r = crate::reduce::to_pairwise(&cubic).unwrap();
        assert!(!Fabric::toshiba_sqbm(Z1_SPICE).check(&r.program).iter()
                    .any(|u| matches!(u, Unsupported::ArityTooHigh { .. })));
        // which the PUBO solver never needed
        assert_eq!(r.ancillas > 0, true, "the reduction really did cost ancillas");
    }

    #[test]
    fn a_fully_connected_fabric_needs_no_embedding() {
        // The distinction native_placement exists for. 100,000 Fujitsu bits ARE 100,000 variables;
        // 5,640 D-Wave qubits are not 5,640 variables.
        // Coefficients that fit BOTH fabrics, so the only thing separating the two answers is
        // placement. D-Wave's couplings live in [-1, 1] and Fujitsu wants whole numbers, so 1 and
        // -1 are the values both accept.
        let p = program(&[1.0, -1.0, 1.0], &[(0, 1.0)]);
        let da = Fabric::fujitsu_da3(Z1_SPICE);
        assert!(da.native_placement, "fully coupled: every bit reaches every other");
        assert!(da.verdict(&p).unwrap().is_runnable(), "so it can promise a run");
        assert!(
            Fabric::dwave_advantage2(Z1_SPICE)
                .verdict(&p)
                .unwrap()
                .caveats()
                .iter()
                .any(|c| matches!(c, Caveat::NeedsEmbedding { .. })),
            "where a sparse hardware graph cannot"
        );
    }

    #[test]
    fn fujitsu_takes_whole_numbers_and_toshiba_does_not_care() {
        // Fujitsu's coefficients are signed integers; the service silently scales and rounds a
        // submission that is not, which is a different program than the one written.
        let fractional = program(&[1.5, 2.0], &[]);
        let da = Fabric::fujitsu_da3(Z1_SPICE);
        let bad = da.check(&fractional);
        assert_eq!(bad.len(), 1, "{bad:?}");
        assert!(bad[0].to_string().contains("integers"), "{}", bad[0]);
        assert!(da.check(&program(&[1.0, 2.0], &[])).is_empty(), "whole numbers are fine");

        // Toshiba's are float32, so 1.5 is exactly what it says
        assert!(Fabric::toshiba_sqbm(Z1_SPICE).check(&fractional).is_empty());
    }

    #[test]
    fn field_bits_is_enforced_and_not_merely_declared() {
        // Every fabric in this file declares it and nothing read it, so a fabric holding three-bit
        // fields accepted a program with fine-grained ones and lost them at submission.
        let mut f = Fabric::unconstrained("coarse-fields", Z1_SPICE);
        f.field_precision = Precision::Fixed { bits: 3 };

        // three signed bits give three levels: a field 1/50th of the peak cannot survive
        let fine = program(&[1.0], &[(0, 4.0), (1, 0.08)]);
        let bad = f.check(&fine);
        assert_eq!(bad.len(), 1, "{bad:?}");
        assert!(matches!(bad[0], Unsupported::CouplingPrecision { bits: 3, .. }), "{:?}", bad[0]);

        // and fields already on the grid survive it
        assert!(f.check(&program(&[1.0], &[(0, 3.0), (1, 1.0)])).is_empty());
    }

    #[test]
    fn the_cpu_refuses_a_higher_order_program_by_declaring_its_arity() {
        // It lowers through to_graph, which is pairwise. It used to pass `check` and then fail
        // afterwards with a HARDCODED arity of 3, reporting three however many the program had.
        let mut cpu = Cpu::default();
        let cubic = Program::from_ftp("ftp 1\nspins 5\nfactor 1 0 1 2 3\n").unwrap();
        let bad = cpu.program(&cubic);
        assert_eq!(bad.len(), 1, "{bad:?}");
        match bad[0] {
            Unsupported::ArityTooHigh { arity, limit } => {
                assert_eq!(arity, 4, "the program's real arity, not a constant");
                assert_eq!(limit, 2);
            }
            ref other => panic!("{other:?}"),
        }
        assert!(cpu.program(&program(&[1.0, 2.0], &[])).is_empty(), "pairwise still runs");
    }

    #[test]
    fn a_unary_factor_is_a_field_here_as_it_is_everywhere_else() {
        // `Program::to_graph` lowers an arity-1 factor with `b.bias(...)`, so it IS a field. This
        // checked it against the COUPLING range and never showed it to the field checks, so a
        // program whose fields were written that way walked past a fabric that has none.
        let src = "ftp 1\nspins 3\nfactor 3.0 0\nfactor 0.5 0 1\n";
        let p = Program::from_ftp(src).unwrap();

        // D-Wave: couplings [-1,1], fields [-4,4]. A unary 3.0 is a legal field and an illegal
        // coupling, so which side it is checked on decides the answer.
        let dw = Fabric::dwave_advantage2(Z1_SPICE);
        assert!(dw.check(&p).is_empty(), "3.0 is a field of 3, which fits [-4, 4]: {:?}", dw.check(&p));

        // and a fabric with no field must see it
        let mut fieldless = Fabric::unconstrained("fieldless", Z1_SPICE);
        fieldless.supports_field = false;
        let bad = fieldless.check(&p);
        assert_eq!(bad.len(), 1, "{bad:?}");
        assert!(matches!(bad[0], Unsupported::NoFieldSupport { nodes: 1 }), "{:?}", bad[0]);
    }

    #[test]
    fn degree_counts_edges_rather_than_factor_mentions() {
        // Two factors on the same pair are ONE edge with a summed weight once the program becomes
        // a graph. Counting the mentions refuses a program the fabric can hold.
        let src = "ftp 1\nspins 3\nfactor 1.0 0 1\nfactor 1.0 0 1\nfactor 1.0 0 2\n";
        let p = Program::from_ftp(src).unwrap();
        assert_eq!(p.to_graph().unwrap().n_edges, 2, "the graph really has two edges");

        let mut f = Fabric::unconstrained("degree-2", Z1_SPICE);
        f.max_degree = Some(2);
        assert!(f.check(&p).is_empty(), "spin 0 has two neighbours, not three: {:?}", f.check(&p));

        f.max_degree = Some(1);
        assert!(!f.check(&p).is_empty(), "and a limit of one really is exceeded");
    }

    #[test]
    fn uniform_couplings_compares_values_not_bit_patterns() {
        // 0.0 and -0.0 are the same coupling and have different bits. Comparing bits reported two
        // distinct weights on a fabric that permits only one.
        let src = "ftp 1\nspins 3\nfactor 0.0 0 1\nfactor -0.0 1 2\n";
        let p = Program::from_ftp(src).unwrap();
        let mut f = Fabric::unconstrained("counting", Z1_SPICE);
        f.uniform_couplings = true;
        assert!(f.check(&p).is_empty(), "one weight, written two ways: {:?}", f.check(&p));

        // and genuinely different weights are still caught
        let two = Program::from_ftp("ftp 1\nspins 3\nfactor 1.0 0 1\nfactor 2.0 1 2\n").unwrap();
        assert!(matches!(f.check(&two)[..], [Unsupported::NonUniformCouplings { distinct: 2 }]));
    }

    #[test]
    fn the_worst_offender_is_reported_because_it_sets_the_scale() {
        // The comment said WORST and the code took the FIRST. Worst is the useful one: it is the
        // value that determines the factor a caller needs.
        let f = Fabric::dwave_advantage2(Z1_SPICE);
        let bad = f.check(&program(&[2.0, 9.0, 3.0], &[]));
        assert_eq!(bad.len(), 1, "{bad:?}");
        match bad[0] {
            Unsupported::OutOfRange { value, .. } => {
                assert_eq!(value, 9.0, "9 is the worst, and 2 merely came first");
            }
            ref other => panic!("{other:?}"),
        }
        // and the factor it implies really does fix the program
        let s = f.scale_to_fit(&program(&[2.0, 9.0, 3.0], &[])).unwrap();
        assert!(f.check(&program(&[2.0 * s, 9.0 * s, 3.0 * s], &[])).is_empty());
    }

    #[test]
    fn a_one_sided_range_does_not_scale_the_program_to_nothing() {
        // The defect this replaced. The factor was computed against min(|lo|, |hi|), which is ZERO
        // for any range with an endpoint at zero -- and the Pt V2's field range is [0, 6]. Every
        // coefficient scaled to zero, and a program of all zeroes "fits" every range there is.
        let mut f = Fabric::unconstrained("one-sided", Z1_SPICE);
        f.field_range = Some(Range::integers(0.0, 6.0));

        let p = program(&[1.0], &[(0, 2.0), (1, 12.0)]);
        let s = f.scale_to_fit(&p).expect("2 and 12 can be scaled into [0, 6]");
        assert!(s > 0.0, "a factor of {s} annihilates the program");
        assert!(f.check(&program(&[1.0], &[(0, 2.0 * s), (1, 12.0 * s)])).is_empty(),
                "and the scaled program really fits");

        // and a negative field genuinely cannot be scaled into [0, 6], which is a None rather than
        // a factor that pretends
        assert_eq!(f.scale_to_fit(&program(&[1.0], &[(0, -1.0)])), None,
                   "no positive factor moves a negative value above zero");
    }

    #[test]
    fn an_asymmetric_range_uses_the_side_each_value_lands_on() {
        // D-Wave's extended_j_range is [-2, 1]. Taking the smaller endpoint magnitude would refuse
        // a -2 the machine holds; taking the larger would permit a +2 it does not.
        let mut f = Fabric::unconstrained("asymmetric", Z1_SPICE);
        f.coupling_range = Some(Range::continuous(-2.0, 1.0));

        assert!(f.check(&program(&[-2.0, 1.0], &[])).is_empty(), "both endpoints are representable");
        assert_eq!(f.check(&program(&[2.0], &[])).len(), 1, "+2 is not");

        // scaling respects the side each value is on: -4 needs 1/2, +4 needs 1/4, so 1/4 wins
        let s = f.scale_to_fit(&program(&[-4.0, 4.0], &[])).unwrap();
        assert_eq!(s, 0.25, "the positive side binds");
        assert!(f.check(&program(&[-4.0 * s, 4.0 * s], &[])).is_empty());
    }

    #[test]
    fn a_returned_factor_always_passes_check() {
        // The factor comes from a division and can land one ulp outside the range it was computed
        // to satisfy, so scale_to_fit verifies against the same predicate check uses. A
        // scale_to_fit whose answer check then rejects is worse than no answer.
        let f = Fabric::dwave_advantage2(Z1_SPICE);
        let mut checked = 0;
        for k in 1..400 {
            let peak = k as f64 * 0.37 + 0.11; // arbitrary magnitudes, none of them round
            let p = program(&[peak, -peak / 3.0, peak / 7.0], &[(0, peak * 2.0)]);
            if let Some(s) = f.scale_to_fit(&p) {
                let scaled = program(
                    &[peak * s, -peak / 3.0 * s, peak / 7.0 * s],
                    &[(0, peak * 2.0 * s)],
                );
                assert!(f.check(&scaled).is_empty(),
                        "peak {peak} scaled by {s} still violates: {:?}", f.check(&scaled));
                checked += 1;
            }
        }
        assert!(checked > 300, "the sweep must actually exercise it: {checked}");
    }

    #[test]
    fn integrality_is_per_range_not_shared() {
        // A fabric whose couplings are whole numbers may still take continuous fields. Requiring
        // both to land on integers would refuse programs the machine accepts.
        let mut f = Fabric::unconstrained("mixed", Z1_SPICE);
        f.coupling_range = Some(Range::integers(-7.0, 7.0));
        f.field_range = Some(Range::continuous(-4.0, 4.0));

        // couplings already whole, field fractional: this fits as written
        assert!(f.check(&program(&[3.0, -2.0], &[(0, 1.5)])).is_empty());
        // and a program needing scaling gets a factor that leaves the couplings whole while the
        // field stays wherever it lands, which is exactly what a mixed fabric allows
        let s = f.scale_to_fit(&program(&[14.0, -28.0], &[(0, 1.5)])).expect("a factor exists");
        assert!(f.check(&program(&[14.0 * s, -28.0 * s], &[(0, 1.5 * s)])).is_empty(),
                "scaled by {s}: {:?}", f.check(&program(&[14.0 * s, -28.0 * s], &[(0, 1.5 * s)])));
        assert_eq!((14.0 * s).fract(), 0.0, "the couplings land on whole numbers");
        assert_ne!((1.5 * s).fract(), 0.0, "and the field is not forced to");
    }

    #[test]
    fn an_integral_fabric_finds_a_smaller_factor_when_the_saturating_one_does_not_land() {
        // The saturating factor rarely puts every value on a whole number. Returning None there
        // would refuse programs that a smaller factor fits exactly.
        let mut f = Fabric::unconstrained("integral", Z1_SPICE);
        f.coupling_range = Some(Range::integers(-7.0, 7.0));

        // The saturating factor is 7/10, which sends 5 to 3.5 and does not land. A smaller one
        // does. The assertion is on the PROPERTIES rather than a number copied from the
        // implementation, which would only restate what the code already does.
        let s = f.scale_to_fit(&program(&[10.0, 5.0], &[])).expect("a smaller factor lands");
        assert!(s > 0.0 && s <= 0.7, "no larger than the saturating factor: {s}");
        assert_eq!((10.0 * s).fract(), 0.0, "10 lands: {}", 10.0 * s);
        assert_eq!((5.0 * s).fract(), 0.0, "5 lands: {}", 5.0 * s);
        assert!(f.check(&program(&[10.0 * s, 5.0 * s], &[])).is_empty());
        // and it is the LARGEST such factor: anything bigger either overflows or does not land
        for bigger in [s * 1.0001, 0.7] {
            let ok = (10.0 * bigger).fract() == 0.0
                && (5.0 * bigger).fract() == 0.0
                && f.check(&program(&[10.0 * bigger, 5.0 * bigger], &[])).is_empty();
            assert!(!ok || bigger <= s, "{bigger} would have been a better answer than {s}");
        }
    }

    #[test]
    fn an_empty_violation_list_does_not_promise_a_run_where_embedding_is_needed() {
        // The distinction this exists for. `check` answers "what rules this out"; on a machine that
        // places variables by minor embedding, nothing ruling it out is not the same as it fitting,
        // and an empty list read as a yes is how someone buys machine time for a program that
        // cannot be placed.
        let p = program(&[0.5, -0.5, 0.25], &[(0, 1.0)]);

        let dw = Fabric::dwave_advantage2(Z1_SPICE);
        assert!(dw.check(&p).is_empty(), "nothing declared rules it out");
        let v = dw.verdict(&p).expect("nothing rules it out");
        assert!(!v.is_runnable(), "a D-Wave part cannot promise a run");
        // BOTH caveats, not whichever was checked first: it embeds AND its precision is unpublished
        assert!(v.caveats().iter().any(|c| matches!(
            c,
            Caveat::NeedsEmbedding { vars, sites } if *vars == p.spins && *sites == Some(4_400)
        )), "{:?}", v.caveats());
        assert!(v.caveats().iter().any(|c| matches!(c, Caveat::LimitsUnstated { .. })),
                "{:?}", v.caveats());
        assert!(v.to_string().contains("do not mean"));

        // A fabric that places one variable per site and publishes its limits CAN promise it.
        let cpu = Fabric::unconstrained("sim", Z1_SPICE);
        assert!(cpu.verdict(&p).unwrap().is_runnable());

        // And a real violation still comes back as one, ahead of any verdict.
        let too_big = program(&[5.0], &[]);
        assert!(matches!(dw.verdict(&too_big), Err(v) if v.len() == 1), "{:?}", dw.verdict(&too_big));
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
        assert_eq!(a2.coupling_precision, Precision::Unstated,
                   "analog, and D-Wave publishes no bit count -- Exact would claim every f64 \
                    arrives intact, which is the opposite of true here");
        assert!(!a2.native_placement, "variables are placed by minor embedding, not one per qubit");

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
            coupling_precision: Precision::Fixed { bits: 8 },
            field_precision: Precision::Fixed { bits: 8 },
            supports_field: false,
            max_arity: 2,
            coupling_range: None,
            field_range: None,
            native_placement: true,
            unstated: &[],
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
