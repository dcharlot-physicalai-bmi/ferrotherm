//! The p-dit: a probabilistic digit of radix `q`, of which the p-bit is the case `q = 2` -- its
//! embodiments, what each costs, the radix economy that says which radix information science
//! prefers, a cycle-exact fixed-point p-trit fabric with its own exact law, and the question the
//! rest of this crate could not ask until now: what a native `q`-state unit buys over `q` p-bits
//! spelling the same variable, measured exactly.
//!
//! # The unit
//!
//! A p-bit is `sign(field + noise)` with logistic noise, which lands on `+1` with probability
//! `sigma(2 beta f)`. A p-dit of radix `q` draws its state from the softmax of its `q` local
//! fields, `P(a) ~ exp(beta f_a)`: the heat-bath update of a `q`-state Potts or clock spin
//! ([`crate::potts`]), and the conditional every categorical Gibbs sampler needs. Three ways to
//! build one, each an [`Embodiment`]:
//!
//! * **Gumbel-max.** Add an independent Gumbel draw `g_a = -ln(-ln u_a)` to each field and take
//!   the argmax. The result is exactly softmax (Gumbel 1954; Maddison, Tarlow and Minka 2014), and
//!   it needs `q` noise draws, `q` adders and a `q`-way argmax: no exponential, no sum, no
//!   division. At `q = 2` the difference of two Gumbels is logistic, so `argmax(f + g_1, g_2)`
//!   IS `sign(f + logistic)`: the p-bit is the two-state Gumbel-max unit, and
//!   `the_p_bit_is_the_two_state_p_dit` holds [`softmax`] against [`crate::kernel::p_up`] to
//!   `1e-15`.
//! * **Cumulative.** One uniform draw, `q` exponentials, a running sum and `q - 1` comparisons --
//!   the inverse-CDF sampler. Fewest draws, but the sum must be normalised, which is a division
//!   or a multiply the p-bit never needed.
//! * **Encoded.** `q` p-bits one-hot, or `q - 1` domain-wall, with a penalty that forbids the
//!   codes that decode to nothing ([`crate::encode`]). This is how every p-bit fabric and every
//!   annealer handles a categorical variable today, and it is the case a native unit is measured
//!   against.
//!
//! # The radix economy
//!
//! Hartley's argument (1928; von Neumann's 1945 draft report makes the same one) charges a digit
//! its radix: `r` symbols to distinguish, so a number of `N` values costs `r log_r N`, and the
//! cost per unit of information is `r / ln r`, which [`radix_economy`] evaluates. It is minimised
//! at `r = e`, and among the integers at `r = 3`: ternary beats binary by 5.7% on this account and
//! quaternary ties binary exactly. Whether a p-trit inherits that advantage depends on what the
//! unit is charged for, and [`bits_per_charge`] states all three cases rather than picking the
//! flattering one:
//!
//! | charged for | binary p-bit | ternary p-trit (Gumbel-max) | verdict |
//! |---|---|---|---|
//! | the radix (Hartley) | `1/2` | `log2(3)/3 = 0.528` | ternary wins by 5.7% |
//! | comparators | `1` | `log2(3)/2 = 0.79` | binary wins |
//! | noise draws | `1` | `log2(3)/3 = 0.528` | binary wins, and the one-hot encoding draws the same `q` |
//!
//! So the radix economy is not where the p-trit's case lies. Its case is that a `q`-state
//! variable spelled in p-bits carries codes that decode to nothing -- `1 - q / 2^q` of them
//! one-hot, `1 - q / 2^(q-1)` domain-wall ([`wasted_codes`]) -- a penalty to keep the chain out
//! of them, which distorts the objective ([`crate::categorical`]), and a chain that still visits
//! them, which is time. A native unit has no invalid state and no penalty, and the exact
//! comparison below says what that is worth.
//!
//! # The fixed-point p-trit, and its floor
//!
//! [`FixedPdit`] is the Gumbel-max unit in the arithmetic of [`crate::hdl`]'s fabric: Q.8 fields
//! with `beta` folded into the weights, a Gumbel ROM of `2^bits` entries indexed by the top bits
//! of a per-node xorshift32, `q` draws per update, an argmax that breaks ties towards the lowest
//! state. Its single-site law is a finite sum, and [`rom_conditional`] evaluates it exactly, so
//! the fabric's law is known without sampling it. That law has a **floor**: a ROM of `b` bits
//! spans Gumbel values from `-ln(ln 2^b)` to `-ln(-ln(1 - 2^-b))`, a range of about `9.7` at ten
//! bits, and a state whose field trails the leader by more than that range has probability
//! **exactly zero** -- the p-dit's version of the comparator floor in [`crate::autocorr`], and like
//! it a function of a bit width, not of `beta`. `examples/pdit_exact.rs` tabulates the total
//! variation from the softmax and the gap where the floor bites for 8 to 16 bits.
//!
//! [`FixedCumulative`] is the other embodiment in the same arithmetic: one draw, an exponential
//! ROM addressed by each state's gap to the leader in steps of `1/64`, sixteen-bit weights summed
//! to `Z`, a `16 x Z` multiply for the threshold `(u Z) >> 16`, and a walk up the cumulative sum.
//! Its exact law is [`cumulative_conditional`], and its floor is set by the **weight width**, not
//! the address width: a weight `round(65535 e^-g)` is zero once `g > 11.78`, which is the same
//! `2 beta f > 11.8` the sixteen-bit comparator p-bit floors at, because it is the same sixteen
//! bits. So the two embodiments floor for different reasons -- Gumbel-max at the ROM's span,
//! cumulative at the weight's resolution -- and cost differently: `q` draws and no multiply
//! against one draw and one multiply.
//!
//! # The comparison, exactly
//!
//! [`native_row`] builds the exact heat-bath sweep kernel over the `q^n` states of a small Potts
//! model; [`encoded_row`] builds the exact sequential-Gibbs sweep kernel over the spins of its
//! one-hot or domain-wall spelling ([`lower`]) at a chosen penalty. Each reports the mixing time
//! to total variation `1/4` from the worst start, the integrated autocorrelation time of one
//! decoded indicator, and the stationary mass on codes that decode to something -- in sweeps and
//! in noise draws, because a sweep of the native unit is `qn` draws, of the one-hot spelling `qn`,
//! of the domain wall `(q-1)n`. Not Kemeny's constant: see [`kemeny_of`] for why that counts
//! states and misled the first draft of this module.
//!
//! On the three-state triangle, `J = 1`, `examples/pdit_exact.rs`:
//!
//! | `beta` | embodiment | penalty | valid mass | mixing, draws | `tau_int`, draws |
//! |---|---|---|---|---|---|
//! | 0.5 | native p-trit | -- | 1 | 9 | 10.6 |
//! | 0.5 | domain wall | 2 | 0.979 | 18 | 12.0 |
//! | 0.5 | one-hot | 4 | 0.548 | 36 | 29.0 |
//! | 2 | native p-trit | -- | 1 | 90 | 152 |
//! | 2 | domain wall | 1 | 0.9993 | 120 | 216 |
//! | 2 | one-hot | 2 | 0.880 | 288 | 481 |
//! | 2 | one-hot | 4 | 0.9978 | 16,416 | 30,333 |
//!
//! Hot, everything mixes in a sweep or two and the native unit's edge is the draws a sweep costs.
//! Cold, the penalty that makes an encoding valid is the penalty that freezes it: the one-hot
//! spelling needs a penalty of four to put 99.8% of its mass on decodable states, and at that
//! penalty it takes **two hundred times** the native unit's draws per independent sample; the
//! domain wall reaches 99.9% valid at a penalty of one and costs 1.4 times the native unit. So a
//! p-trit's advantage over p-bits is not the radix economy and not the spin count. It is that the
//! penalty is gone, and with it the frozen chain the penalty buys. Measured before it was written
//! down, in `the_penalty_that_makes_one_hot_valid_freezes_it_and_the_native_unit_needs_none`.
//!
//! # What the embodiments cost in cells
//!
//! `examples/pdit_synth.rs` runs yosys's generic flow over sixteen sites of each fabric, weights
//! baked into the netlist as [`crate::hdl`] bakes them:
//!
//! | unit | cells per unit | what dominates |
//! |---|---|---|
//! | p-bit, ten-bit sigmoid ROM | 173 | the ROM folds to the few field values a fixed graph reaches |
//! | cumulative p-trit, `q = 3` | 1,676 | one `16 x Z` multiply; the exponential ROM folds like the sigmoid |
//! | Gumbel-max p-trit, `q = 3`, ten-bit ROM | 4,027 | three Gumbel ROMs, addressed by noise, which cannot fold |
//!
//! Generic cells are a relative measure and the ratio is the result: the embodiment with no
//! multiply pays for it in three noise-addressed ROMs that synthesis cannot shrink, and the one
//! with a multiply pays for the multiplier, which a board with DSP slices would price differently.
//! Per three-state variable that is one unit against three one-hot p-bits (519 cells and three
//! penalty couplings) or two domain-wall p-bits (346 and one) -- so on cells alone the p-bits win,
//! and the p-trit's case stays where the exact comparison put it: the chain that does not freeze.
//!
//! # What this is not
//!
//! A p-trit here is a Gumbel-max or cumulative unit in Q.8 arithmetic with an emulator and
//! synthesizable RTL that agree bit for bit under icarus-verilog. Nobody has metered one: a joules figure for it is
//! the ledger's p-bit price times the draws until a board says otherwise, and the ROM floor above
//! is a property of the RTL as emitted, not of any silicon.

use crate::autocorr::{self, Kernel};
use crate::encode::{Encoding, Slot};
use crate::graph::{Graph, GraphBuilder};
use crate::hdl::FRAC;
use crate::potts::{Interaction, Potts};

/// The most states a dense sweep kernel is built over: `q^n` up to this, `729 = 3^6`.
pub const MAX_DENSE_STATES: usize = 729;

/// Address width of the Gumbel ROM the fixed-point unit indexes by default: `2^10` entries.
pub const DEFAULT_ROM_BITS: u32 = 10;

/// How a `q`-valued stochastic variable is realised.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Embodiment {
    /// `sign(field + logistic noise)`: the p-bit, `q = 2` only.
    Comparator,
    /// `argmax_a (field_a + Gumbel_a)`: the native p-dit, exactly softmax.
    GumbelMax,
    /// One uniform against the running sum of `q` exponentials: inverse CDF, with a normalisation.
    Cumulative,
    /// `q` p-bits, exactly one hot, held there by an all-to-all penalty.
    OneHot,
    /// `q - 1` p-bits holding one domain wall, held there by a chain penalty.
    DomainWall,
}

/// What one update of one `q`-valued variable costs under an embodiment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cost {
    /// Stochastic units (p-bits or p-dits) the variable occupies.
    pub units: usize,
    /// Independent noise draws per update.
    pub draws: usize,
    /// Comparators per update.
    pub comparators: usize,
    /// Adders on the noise path per update (the field sums are common to every embodiment).
    pub adders: usize,
    /// ROM reads per update (sigmoid, Gumbel or exponential).
    pub rom_reads: usize,
    /// Divisions (or normalising multiplies) per update.
    pub divisions: usize,
    /// Penalty couplings the embodiment drags into the model.
    pub penalty_couplings: usize,
    /// Whether every state the unit can be in decodes to a value.
    pub valid_always: bool,
}

impl Embodiment {
    /// Whether this embodiment can realise a `q`-valued variable at all.
    #[must_use]
    pub fn supports(self, q: usize) -> bool {
        match self {
            Embodiment::Comparator => q == 2,
            _ => q >= 2,
        }
    }

    /// The cost of one update of one `q`-valued variable.
    ///
    /// # Panics
    ///
    /// If the embodiment does not support `q`: a comparator is a two-state device.
    #[must_use]
    pub fn cost(self, q: usize) -> Cost {
        assert!(
            self.supports(q),
            "{} cannot realise a {q}-valued variable",
            self.label()
        );
        match self {
            Embodiment::Comparator => Cost {
                units: 1,
                draws: 1,
                comparators: 1,
                adders: 0,
                rom_reads: 1,
                divisions: 0,
                penalty_couplings: 0,
                valid_always: true,
            },
            Embodiment::GumbelMax => Cost {
                units: 1,
                draws: q,
                comparators: q - 1,
                adders: q,
                rom_reads: q,
                divisions: 0,
                penalty_couplings: 0,
                valid_always: true,
            },
            Embodiment::Cumulative => Cost {
                units: 1,
                draws: 1,
                comparators: q - 1,
                adders: q - 1,
                rom_reads: q,
                divisions: 1,
                penalty_couplings: 0,
                valid_always: true,
            },
            Embodiment::OneHot => Cost {
                units: q,
                draws: q,
                comparators: q,
                adders: 0,
                rom_reads: q,
                divisions: 0,
                penalty_couplings: Encoding::OneHot.penalty_couplings(q),
                valid_always: false,
            },
            Embodiment::DomainWall => Cost {
                units: q - 1,
                draws: q - 1,
                comparators: q - 1,
                adders: 0,
                rom_reads: q - 1,
                divisions: 0,
                penalty_couplings: Encoding::DomainWall.penalty_couplings(q),
                valid_always: false,
            },
        }
    }

    /// The spin encoding an encoded embodiment lowers to, `None` for a native unit.
    #[must_use]
    pub fn encoding(self) -> Option<Encoding> {
        match self {
            Embodiment::OneHot => Some(Encoding::OneHot),
            Embodiment::DomainWall => Some(Encoding::DomainWall),
            _ => None,
        }
    }

    /// A short label, for tables.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Embodiment::Comparator => "comparator p-bit",
            Embodiment::GumbelMax => "Gumbel-max p-dit",
            Embodiment::Cumulative => "cumulative p-dit",
            Embodiment::OneHot => "one-hot p-bits",
            Embodiment::DomainWall => "domain-wall p-bits",
        }
    }
}

/// What a unit is charged for, when its information yield is priced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Charge {
    /// Hartley's account: a digit costs its radix.
    Radix,
    /// The comparators one update needs.
    Comparators,
    /// The independent noise draws one update needs.
    Draws,
}

/// Hartley's cost per unit of information of a digit of radix `r`: `r / ln r`, minimised at
/// `r = e`.
///
/// # Panics
///
/// If `r` is not above one: a one-symbol digit carries nothing.
#[must_use]
pub fn radix_economy(r: f64) -> f64 {
    assert!(r > 1.0, "a radix must exceed one, got {r}");
    r / r.ln()
}

/// The integer radix with the lowest [`radix_economy`], searched over `2..=64`: three.
#[must_use]
pub fn best_integer_radix() -> usize {
    let mut best = 2usize;
    for r in 3..=64usize {
        if radix_economy(r as f64) < radix_economy(best as f64) {
            best = r;
        }
    }
    best
}

/// Bits of information one update of a `q`-valued variable can carry, `log2 q`, per unit of
/// what it is charged for under `charge`.
///
/// # Panics
///
/// As [`Embodiment::cost`].
#[must_use]
pub fn bits_per_charge(q: usize, embodiment: Embodiment, charge: Charge) -> f64 {
    let cost = embodiment.cost(q);
    let charged = match charge {
        Charge::Radix => q,
        Charge::Comparators => cost.comparators,
        Charge::Draws => cost.draws,
    };
    (q as f64).log2() / charged as f64
}

/// The fraction of an encoding's spin codes that decode to no value: `1 - q / 2^spins`.
///
/// # Panics
///
/// If `q` is below 2.
#[must_use]
pub fn wasted_codes(encoding: Encoding, q: usize) -> f64 {
    1.0 - q as f64 / 2f64.powi(encoding.spins(q) as i32)
}

/// The p-dit's law: `P(a) ~ exp(beta f_a)` over the fields, computed with the largest exponent
/// shifted out so no field overflows it.
#[must_use]
pub fn softmax(fields: &[f64], beta: f64) -> Vec<f64> {
    let top = fields.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let w: Vec<f64> = fields.iter().map(|&f| (beta * (f - top)).exp()).collect();
    let z: f64 = w.iter().sum();
    w.iter().map(|x| x / z).collect()
}

/// The Gumbel ROM: `-ln(-ln u)` at `u = (a + 0.5) / 2^bits` for every address `a`, in Q.8,
/// ascending in `a`.
///
/// # Panics
///
/// If `bits` is outside `4..=16`: below four the ROM is too coarse to mean anything, above sixteen
/// the address no longer fits the top half of a 32-bit draw.
#[must_use]
pub fn gumbel_rom(bits: u32) -> Vec<i32> {
    assert!(
        (4..=16).contains(&bits),
        "a Gumbel ROM takes 4 to 16 address bits, not {bits}"
    );
    let scale = f64::from(1u32 << FRAC);
    let entries = 1usize << bits;
    (0..entries)
        .map(|a| {
            let u = (a as f64 + 0.5) / entries as f64;
            let g = -(-u.ln()).ln();
            (g * scale).round() as i32
        })
        .collect()
}

/// The exact law of `argmax_a (fields_q[a] + g_a)` when each `g_a` is drawn uniformly from the
/// ROM, ties going to the lowest state.
///
/// For each state `a` and each ROM value `v` it can draw, the states below `a` must draw strictly
/// less than `fields_q[a] + v - fields_q[b]` and the states above at most that, and the ROM is
/// sorted, so each of those is a partition point. `O(q^2 2^bits)`, and a probability of exactly
/// zero is a real zero: no draw in the ROM can make that state win.
///
/// # Panics
///
/// If the ROM is empty or `fields_q` is.
#[must_use]
pub fn rom_conditional(fields_q: &[i32], rom: &[i32]) -> Vec<f64> {
    assert!(
        !rom.is_empty() && !fields_q.is_empty(),
        "a conditional needs a ROM and at least one field"
    );
    let q = fields_q.len();
    let entries = rom.len() as f64;
    let mut law = vec![0.0f64; q];
    for a in 0..q {
        let mut mass = 0.0;
        for &v in rom {
            let score = fields_q[a] + v;
            let mut chance = 1.0;
            for b in 0..q {
                if b == a {
                    continue;
                }
                // g_b must satisfy fields_q[b] + g_b < score (b below a) or <= score (b above a).
                let bound = score - fields_q[b];
                let count = if b < a {
                    rom.partition_point(|&g| g < bound)
                } else {
                    rom.partition_point(|&g| g <= bound)
                };
                chance *= count as f64 / entries;
                if chance == 0.0 {
                    break;
                }
            }
            mass += chance;
        }
        law[a] = mass / entries;
    }
    law
}

/// The span of a Gumbel ROM in field units: the largest field gap a state can still overcome.
#[must_use]
pub fn rom_span(rom: &[i32]) -> f64 {
    let scale = f64::from(1u32 << FRAC);
    let lo = rom.iter().copied().min().unwrap_or(0);
    let hi = rom.iter().copied().max().unwrap_or(0);
    f64::from(hi - lo) / scale
}

fn xorshift32(mut x: u32) -> u32 {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    x
}

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = z;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Why a model cannot be laid into the fixed-point fabric or a dense kernel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unfit {
    /// The fabric carries Potts couplings only; a clock table is a later revision.
    NotPotts,
    /// `q^n` states exceed [`MAX_DENSE_STATES`].
    TooManyStates {
        /// States per site.
        q: usize,
        /// Sites.
        n: usize,
    },
    /// The chain has more than one closed class, so no unique stationary law.
    Reducible,
}

impl core::fmt::Display for Unfit {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Unfit::NotPotts => write!(
                f,
                "the fixed-point p-dit fabric carries Potts couplings only"
            ),
            Unfit::TooManyStates { q, n } => write!(
                f,
                "{q}^{n} states exceed the dense limit ({MAX_DENSE_STATES} native, 2^{MAX_ENCODED_SPINS} encoded)"
            ),
            Unfit::Reducible => write!(f, "the chain has more than one closed class"),
        }
    }
}

impl core::error::Error for Unfit {}

/// Adjacency of a Potts model as `(neighbour, J)` per site, each undirected edge on both ends.
fn adjacency(m: &Potts) -> Vec<Vec<(usize, f64)>> {
    let mut adj = vec![Vec::new(); m.n()];
    for (i, j, w) in m.edges() {
        adj[i].push((j, w));
        adj[j].push((i, w));
    }
    adj
}

/// A greedy proper colouring: each site takes the lowest colour none of its earlier neighbours
/// holds. Any proper colouring makes a class-parallel sweep equal to a sequential one.
fn colour_classes(adj: &[Vec<(usize, f64)>]) -> Vec<Vec<u32>> {
    let n = adj.len();
    let mut colour = vec![usize::MAX; n];
    let mut classes: Vec<Vec<u32>> = Vec::new();
    for i in 0..n {
        let mut used = vec![false; classes.len() + 1];
        for &(j, _) in &adj[i] {
            if colour[j] != usize::MAX {
                used[colour[j]] = true;
            }
        }
        let c = used.iter().position(|&u| !u).unwrap_or(classes.len());
        if c == classes.len() {
            classes.push(Vec::new());
        }
        colour[i] = c;
        classes[c].push(i as u32);
    }
    classes
}

/// A cycle-exact fixed-point emulator of a Gumbel-max p-dit fabric over a Potts model.
///
/// Same conventions as [`crate::hdl::FixedFabric`]: Q.8 weights with `beta` folded in, one
/// xorshift32 per node, a ROM indexed by the top bits of a draw, colour classes updated one per
/// phase. The one new thing is the argmax over `q` scores, which is `q - 1` comparators.
#[derive(Clone, Debug)]
pub struct FixedPdit {
    /// States per site.
    pub q: usize,
    /// Sites.
    pub n: usize,
    /// Per site: `(neighbour, Q.8 weight of beta J)`.
    pub adj: Vec<Vec<(u32, i32)>>,
    /// Per site and state, `n * q` entries: Q.8 of `beta h_i(a)`.
    pub bias_q: Vec<i32>,
    /// Colour classes, one updated per phase.
    pub classes: Vec<Vec<u32>>,
    /// The Gumbel ROM, Q.8, ascending.
    pub rom: Vec<i32>,
    /// Address bits of the ROM.
    pub rom_bits: u32,
    /// Per-node RNG seeds.
    pub seeds: Vec<u32>,
    /// Power-on state.
    pub init_s: Vec<u8>,
    /// Current state.
    pub s: Vec<u8>,
    /// Per-node xorshift32 state, advanced `q` times per update.
    pub rng: Vec<u32>,
}

impl FixedPdit {
    /// Quantise `m` at inverse temperature `beta` into a fabric with a `rom_bits`-bit Gumbel ROM.
    ///
    /// # Errors
    ///
    /// [`Unfit::NotPotts`] for a clock model.
    ///
    /// # Panics
    ///
    /// As [`gumbel_rom`].
    pub fn new(m: &Potts, beta: f64, seed: u64, rom_bits: u32) -> Result<FixedPdit, Unfit> {
        if m.kind() != Interaction::Potts {
            return Err(Unfit::NotPotts);
        }
        let scale = f64::from(1u32 << FRAC);
        let q = m.q();
        let n = m.n();
        let adj = adjacency(m)
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&(j, w)| (j as u32, (beta * w * scale).round() as i32))
                    .collect()
            })
            .collect();
        let mut bias_q = vec![0i32; n * q];
        for i in 0..n {
            for a in 0..q {
                bias_q[i * q + a] = (beta * m.field(i, a as u8) * scale).round() as i32;
            }
        }
        let classes = colour_classes(&adjacency(m));
        let seeds: Vec<u32> = (0..n)
            .map(|i| {
                let s = splitmix(seed ^ (i as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)) as u32;
                if s == 0 {
                    1
                } else {
                    s
                }
            })
            .collect();
        let init_s: Vec<u8> = (0..n)
            .map(|i| (splitmix(seed ^ 0xA5A5 ^ i as u64) % q as u64) as u8)
            .collect();
        Ok(FixedPdit {
            q,
            n,
            adj,
            bias_q,
            classes,
            rom: gumbel_rom(rom_bits),
            rom_bits,
            seeds: seeds.clone(),
            init_s: init_s.clone(),
            s: init_s,
            rng: seeds,
        })
    }

    /// The `q` Q.8 fields of site `i` given a state: `sum_j w_ij [s_j = a] + bias(i, a)`.
    #[must_use]
    pub fn fields_q_of(&self, i: usize, s: &[u8]) -> Vec<i32> {
        let mut f: Vec<i32> = self.bias_q[i * self.q..(i + 1) * self.q].to_vec();
        for &(j, w) in &self.adj[i] {
            f[usize::from(s[j as usize])] += w;
        }
        f
    }

    /// The exact law of site `i`'s next state given the current one, from the ROM.
    #[must_use]
    pub fn conditional_of(&self, i: usize, s: &[u8]) -> Vec<f64> {
        rom_conditional(&self.fields_q_of(i, s), &self.rom)
    }

    fn update_node(&mut self, i: usize) {
        let fields = self.fields_q_of(i, &self.s);
        let shift = 32 - self.rom_bits;
        let mut best = 0usize;
        let mut best_score = i32::MIN;
        for (a, &f) in fields.iter().enumerate() {
            let nx = xorshift32(self.rng[i]);
            self.rng[i] = nx;
            let score = f + self.rom[(nx >> shift) as usize];
            if score > best_score {
                best_score = score;
                best = a;
            }
        }
        self.s[i] = best as u8;
    }

    /// One sweep: every colour class in turn, every site in it updated from the registered state.
    pub fn sweep(&mut self) {
        for c in 0..self.classes.len() {
            let class = self.classes[c].clone();
            for &iu in &class {
                self.update_node(iu as usize);
            }
        }
    }

    /// Back to the power-on state and seeds.
    pub fn reset(&mut self) {
        self.s.copy_from_slice(&self.init_s);
        self.rng.copy_from_slice(&self.seeds);
    }

    /// The sites in the order one sweep updates them: class by class.
    #[must_use]
    pub fn site_order(&self) -> Vec<usize> {
        self.classes.iter().flatten().map(|&i| i as usize).collect()
    }

    /// The exact sweep kernel of this fabric over all `q^n` states, row-major, indexed as
    /// [`Potts::index_of`] does.
    ///
    /// # Errors
    ///
    /// [`Unfit::TooManyStates`] above [`MAX_DENSE_STATES`].
    pub fn sweep_kernel(&self) -> Result<Vec<f64>, Unfit> {
        let order = self.site_order();
        dense_sweep_kernel(self.q, self.n, &order, |i, s| self.conditional_of(i, s))
    }

    /// Bits one site's state occupies in the packed state vector: enough for `q - 1`.
    #[must_use]
    pub fn state_bits(&self) -> usize {
        ((usize::BITS - (self.q - 1).leading_zeros()) as usize).max(1)
    }

    /// Bits the phase counter occupies: enough for the number of colour classes less one.
    #[must_use]
    pub fn phase_bits(&self) -> usize {
        ((usize::BITS - (self.classes.len() - 1).leading_zeros()) as usize).max(1)
    }

    /// The current state packed as the RTL's `state` vector, site `i` at bits `i * sb ..`, as a
    /// hex string of the width the testbench reads.
    #[must_use]
    pub fn packed_hex(&self) -> String {
        let sb = self.state_bits();
        let total = self.n * sb;
        let hexw = total.div_ceil(4);
        let mut val = vec![0u8; hexw];
        for (i, &v) in self.s.iter().enumerate() {
            for b in 0..sb {
                if (v >> b) & 1 == 1 {
                    let bit = i * sb + b;
                    val[hexw - 1 - bit / 4] |= 1 << (bit % 4);
                }
            }
        }
        val.iter().map(|b| format!("{b:x}")).collect()
    }

    /// Emit the synthesizable p-dit fabric: no vendor primitives, one module.
    ///
    /// Per node and per state a Q.8 field summed from the registered neighbour states, `q`
    /// chained xorshift32 draws, `q` Gumbel ROM reads, `q` signed adders and a `q`-way argmax that
    /// breaks ties towards the lowest state; one colour class updated per phase.
    #[must_use]
    pub fn emit_verilog(&self, module: &str) -> String {
        let (n, q) = (self.n, self.q);
        let sb = self.state_bits();
        let pb = self.phase_bits();
        let bits = self.rom_bits;
        let classes = self.classes.len();
        let mut v = String::new();
        v.push_str(&format!(
            "// generated by ferrotherm::pdit -- fixed-point Gumbel-max p-dit fabric\n\
             // {n} p-dits of radix {q}, Q.{FRAC} weights, {}-entry Gumbel ROM, xorshift32 per node, {q} draws per update\n\
             module {module} (\n    input wire clk,\n    input wire rst,\n    input wire en,\n    output reg [{top}:0] state,\n    output reg [{ptop}:0] phase\n);\n",
            1usize << bits,
            top = n * sb - 1,
            ptop = pb - 1
        ));
        v.push_str(&format!(
            "  function signed [15:0] gum; input [{}:0] a; begin\n    case (a)\n",
            bits - 1
        ));
        for (a, &g) in self.rom.iter().enumerate() {
            let lit = if g < 0 {
                format!("-16'sd{}", -g)
            } else {
                format!("16'sd{g}")
            };
            v.push_str(&format!("      {bits}'d{a}: gum = {lit};\n"));
        }
        v.push_str("      default: gum = 16'sd0;\n    endcase\n  end endfunction\n\n");
        v.push_str(
            "  function [31:0] xs32; input [31:0] x; reg [31:0] a, b; begin\n    a = x ^ (x << 13); b = a ^ (a >> 17); xs32 = b ^ (b << 5);\n  end endfunction\n\n",
        );
        v.push_str(&format!("  reg [31:0] rng [0:{}];\n", n - 1));
        for i in 0..n {
            v.push_str(&format!(
                "  wire [{}:0] s{i} = state[{}:{}];\n",
                sb - 1,
                i * sb + sb - 1,
                i * sb
            ));
        }
        for i in 0..n {
            for a in 0..q {
                let bias = self.bias_q[i * q + a];
                let mut terms = vec![if bias < 0 {
                    format!("-32'sd{}", -bias)
                } else {
                    format!("32'sd{bias}")
                }];
                for &(j, w) in &self.adj[i] {
                    let lit = if w < 0 {
                        format!("-32'sd{}", -w)
                    } else {
                        format!("32'sd{w}")
                    };
                    terms.push(format!("(s{j} == {sb}'d{a} ? {lit} : 32'sd0)"));
                }
                v.push_str(&format!(
                    "  wire signed [31:0] f{i}_{a} = {};\n",
                    terms.join(" + ")
                ));
            }
            for a in 0..q {
                let from = if a == 0 {
                    format!("rng[{i}]")
                } else {
                    format!("d{i}_{}", a - 1)
                };
                v.push_str(&format!("  wire [31:0] d{i}_{a} = xs32({from});\n"));
                v.push_str(&format!(
                    "  wire signed [31:0] c{i}_{a} = f{i}_{a} + gum(d{i}_{a}[31:{}]);\n",
                    32 - bits
                ));
            }
            v.push_str(&format!(
                "  wire signed [31:0] b{i}_0 = c{i}_0;\n  wire [{}:0] k{i}_0 = {sb}'d0;\n",
                sb - 1
            ));
            for a in 1..q {
                v.push_str(&format!(
                    "  wire signed [31:0] b{i}_{a} = (c{i}_{a} > b{i}_{p}) ? c{i}_{a} : b{i}_{p};\n  wire [{}:0] k{i}_{a} = (c{i}_{a} > b{i}_{p}) ? {sb}'d{a} : k{i}_{p};\n",
                    sb - 1,
                    p = a - 1
                ));
            }
        }
        v.push_str(&format!(
            "\n  always @(posedge clk) begin\n    if (rst) begin\n      phase <= {pb}'d0;\n"
        ));
        for i in 0..n {
            v.push_str(&format!("      rng[{i}] <= 32'd{};\n", self.seeds[i]));
            v.push_str(&format!(
                "      state[{}:{}] <= {sb}'d{};\n",
                i * sb + sb - 1,
                i * sb,
                self.init_s[i]
            ));
        }
        v.push_str(&format!(
            "    end else if (en) begin\n      phase <= (phase == {pb}'d{}) ? {pb}'d0 : phase + {pb}'d1;\n",
            classes - 1
        ));
        for (ci, class) in self.classes.iter().enumerate() {
            v.push_str(&format!("      if (phase == {pb}'d{ci}) begin\n"));
            for &iu in class {
                let i = iu as usize;
                v.push_str(&format!(
                    "        state[{}:{}] <= k{i}_{}; rng[{i}] <= d{i}_{};\n",
                    i * sb + sb - 1,
                    i * sb,
                    q - 1,
                    q - 1
                ));
            }
            v.push_str("      end\n");
        }
        v.push_str("    end\n  end\nendmodule\n");
        v
    }

    /// A self-checking testbench and the per-sweep packed state trace the emulator produces from
    /// power-on; the testbench prints `FERROTHERM_PASS` only on a bit-exact match over every sweep.
    pub fn emit_testbench(&mut self, module: &str, sweeps: usize) -> (String, String) {
        self.reset();
        let mut expected = String::new();
        for _ in 0..sweeps {
            self.sweep();
            expected.push_str(&self.packed_hex());
            expected.push('\n');
        }
        self.reset();
        let top = self.n * self.state_bits() - 1;
        let ptop = self.phase_bits() - 1;
        let classes = self.classes.len();
        let tb = format!(
            "`timescale 1ns/1ps\nmodule tb;\n  reg clk = 0, rst = 1;\n  wire [{top}:0] state;\n  wire [{ptop}:0] phase;\n  {module} dut(.clk(clk), .rst(rst), .en(1'b1), .state(state), .phase(phase));\n  reg [{top}:0] expected [0:{last}];\n  integer sw, errors = 0;\n  always #5 clk = ~clk;\n  initial begin\n    $readmemh(\"expected.hex\", expected);\n    @(posedge clk); @(posedge clk); @(negedge clk); rst = 0;\n    for (sw = 0; sw < {sweeps}; sw = sw + 1) begin\n      repeat ({classes}) @(posedge clk);\n      #1;\n      if (state !== expected[sw]) begin\n        errors = errors + 1;\n        $display(\"MISMATCH sweep %0d: got %h want %h\", sw, state, expected[sw]);\n      end\n    end\n    if (errors == 0) $display(\"FERROTHERM_PASS\");\n    else $display(\"FERROTHERM_FAIL %0d\", errors);\n    $finish;\n  end\nendmodule\n",
            last = sweeps - 1,
        );
        (tb, expected)
    }
}

/// Weight width of the cumulative unit: sixteen bits, `round(65535 e^-g)`.
pub const WEIGHT_BITS: u32 = 16;

/// The gap step of the exponential ROM: an address is the Q.8 gap shifted right by two, so one
/// address step is `4/256 = 1/64` in field units.
pub const EXP_GAP_SHIFT: u32 = 2;

/// The exponential ROM: `round(65535 e^(-a / 64))` for every address `a`, descending.
///
/// # Panics
///
/// If `bits` is outside `10..=16`: below ten the ROM ends before the weight floor at a gap of
/// `11.78`, and the saturated tail would then carry a weight the exponential does not.
#[must_use]
pub fn exp_rom(bits: u32) -> Vec<u32> {
    assert!(
        (10..=16).contains(&bits),
        "an exponential ROM takes 10 to 16 address bits, not {bits}"
    );
    let entries = 1usize << bits;
    let top = f64::from((1u32 << WEIGHT_BITS) - 1);
    (0..entries)
        .map(|a| {
            let gap = a as f64 * f64::from(1u32 << EXP_GAP_SHIFT) / f64::from(1u32 << FRAC);
            (top * (-gap).exp()).round() as u32
        })
        .collect()
}

/// The sixteen-bit weights of each state from Q.8 fields: the leader gets the top weight, every
/// other state the ROM entry at its gap, saturating at the last address.
#[must_use]
pub fn cumulative_weights(fields_q: &[i32], rom: &[u32]) -> Vec<u32> {
    let top = fields_q.iter().copied().max().unwrap_or(0);
    let last = rom.len() - 1;
    fields_q
        .iter()
        .map(|&f| {
            let gap = (top - f) as u32 >> EXP_GAP_SHIFT;
            rom[(gap as usize).min(last)]
        })
        .collect()
}

/// The exact law of the cumulative unit: with weights `w`, cumulative sums `C`, total `Z` and a
/// sixteen-bit uniform `u`, the state is the first `a` with `(u Z) >> 16 < C_a`, so state `a`
/// owns the draws `u` in `[ceil(C_(a-1) 2^16 / Z), ceil(C_a 2^16 / Z))` -- a count, not an
/// integral, so a zero is a real zero.
///
/// # Panics
///
/// If the ROM or the fields are empty.
#[must_use]
pub fn cumulative_conditional(fields_q: &[i32], rom: &[u32]) -> Vec<f64> {
    assert!(
        !rom.is_empty() && !fields_q.is_empty(),
        "a conditional needs a ROM and at least one field"
    );
    let w = cumulative_weights(fields_q, rom);
    let z: u64 = w.iter().map(|&x| u64::from(x)).sum();
    let draws = 1u64 << WEIGHT_BITS;
    let first_draw_at = |c: u64| (c * draws).div_ceil(z).min(draws);
    let mut law = Vec::with_capacity(w.len());
    let mut below = 0u64;
    for &wa in &w {
        let above = below + u64::from(wa);
        let count = first_draw_at(above) - first_draw_at(below);
        law.push(count as f64 / draws as f64);
        below = above;
    }
    law
}

/// The field gap beyond which a sixteen-bit weight rounds to zero: `ln(2 (2^16 - 1))`.
#[must_use]
pub fn cumulative_floor() -> f64 {
    (2.0 * f64::from((1u32 << WEIGHT_BITS) - 1)).ln()
}

/// A cycle-exact fixed-point emulator of a cumulative (inverse-CDF) p-dit fabric over a Potts
/// model: the same fields, RNG and colour classes as [`FixedPdit`], one draw per update, an
/// exponential ROM, a multiply and a walk.
#[derive(Clone, Debug)]
pub struct FixedCumulative {
    /// The Gumbel-max fabric this shares its fields, seeds and classes with.
    pub base: FixedPdit,
    /// The exponential ROM, sixteen-bit weights, descending.
    pub rom: Vec<u32>,
    /// Address bits of the ROM.
    pub rom_bits: u32,
}

impl FixedCumulative {
    /// Quantise `m` at inverse temperature `beta` with a `rom_bits`-bit exponential ROM.
    ///
    /// # Errors
    ///
    /// As [`FixedPdit::new`].
    ///
    /// # Panics
    ///
    /// As [`exp_rom`].
    pub fn new(m: &Potts, beta: f64, seed: u64, rom_bits: u32) -> Result<FixedCumulative, Unfit> {
        let base = FixedPdit::new(m, beta, seed, DEFAULT_ROM_BITS)?;
        Ok(FixedCumulative {
            base,
            rom: exp_rom(rom_bits),
            rom_bits,
        })
    }

    /// The exact law of site `i`'s next state given the current one.
    #[must_use]
    pub fn conditional_of(&self, i: usize, s: &[u8]) -> Vec<f64> {
        cumulative_conditional(&self.base.fields_q_of(i, s), &self.rom)
    }

    fn update_node(&mut self, i: usize) {
        let w = cumulative_weights(&self.base.fields_q_of(i, &self.base.s), &self.rom);
        let z: u64 = w.iter().map(|&x| u64::from(x)).sum();
        let nx = xorshift32(self.base.rng[i]);
        self.base.rng[i] = nx;
        let u = u64::from(nx >> 16);
        let threshold = (u * z) >> WEIGHT_BITS;
        let mut cumulative = 0u64;
        let mut chosen = w.len() - 1;
        for (a, &wa) in w.iter().enumerate() {
            cumulative += u64::from(wa);
            if threshold < cumulative {
                chosen = a;
                break;
            }
        }
        self.base.s[i] = chosen as u8;
    }

    /// One sweep, class by class.
    pub fn sweep(&mut self) {
        for c in 0..self.base.classes.len() {
            let class = self.base.classes[c].clone();
            for &iu in &class {
                self.update_node(iu as usize);
            }
        }
    }

    /// Back to the power-on state and seeds.
    pub fn reset(&mut self) {
        self.base.reset();
    }

    /// The exact sweep kernel over all `q^n` states.
    ///
    /// # Errors
    ///
    /// [`Unfit::TooManyStates`] above [`MAX_DENSE_STATES`].
    pub fn sweep_kernel(&self) -> Result<Vec<f64>, Unfit> {
        let order = self.base.site_order();
        dense_sweep_kernel(self.base.q, self.base.n, &order, |i, s| {
            self.conditional_of(i, s)
        })
    }

    /// Emit the synthesizable cumulative p-dit fabric: per node the `q` fields, a max tree, `q`
    /// gap-addressed ROM reads, their sum, one draw, one `16 x Z` multiply and a cumulative walk.
    #[must_use]
    pub fn emit_verilog(&self, module: &str) -> String {
        let b = &self.base;
        let (n, q) = (b.n, b.q);
        let sb = b.state_bits();
        let pb = b.phase_bits();
        let bits = self.rom_bits;
        let classes = b.classes.len();
        let last = (1u32 << bits) - 1;
        let mut v = String::new();
        v.push_str(&format!(
            "// generated by ferrotherm::pdit -- fixed-point cumulative (inverse-CDF) p-dit fabric\n\
             // {n} p-dits of radix {q}, Q.{FRAC} weights, {}-entry exponential ROM, xorshift32 per node, one draw per update\n\
             module {module} (\n    input wire clk,\n    input wire rst,\n    input wire en,\n    output reg [{top}:0] state,\n    output reg [{ptop}:0] phase\n);\n",
            1usize << bits,
            top = n * sb - 1,
            ptop = pb - 1
        ));
        v.push_str(&format!(
            "  function [15:0] ex; input [{}:0] a; begin\n    case (a)\n",
            bits - 1
        ));
        for (a, &w) in self.rom.iter().enumerate() {
            v.push_str(&format!("      {bits}'d{a}: ex = 16'd{w};\n"));
        }
        v.push_str("      default: ex = 16'd0;\n    endcase\n  end endfunction\n\n");
        v.push_str(
            "  function [31:0] xs32; input [31:0] x; reg [31:0] a, b; begin\n    a = x ^ (x << 13); b = a ^ (a >> 17); xs32 = b ^ (b << 5);\n  end endfunction\n\n",
        );
        v.push_str(&format!("  reg [31:0] rng [0:{}];\n", n - 1));
        for i in 0..n {
            v.push_str(&format!(
                "  wire [{}:0] s{i} = state[{}:{}];\n",
                sb - 1,
                i * sb + sb - 1,
                i * sb
            ));
        }
        for i in 0..n {
            for a in 0..q {
                let bias = b.bias_q[i * q + a];
                let mut terms = vec![if bias < 0 {
                    format!("-32'sd{}", -bias)
                } else {
                    format!("32'sd{bias}")
                }];
                for &(j, w) in &b.adj[i] {
                    let lit = if w < 0 {
                        format!("-32'sd{}", -w)
                    } else {
                        format!("32'sd{w}")
                    };
                    terms.push(format!("(s{j} == {sb}'d{a} ? {lit} : 32'sd0)"));
                }
                v.push_str(&format!(
                    "  wire signed [31:0] f{i}_{a} = {};\n",
                    terms.join(" + ")
                ));
            }
            v.push_str(&format!("  wire signed [31:0] m{i}_0 = f{i}_0;\n"));
            for a in 1..q {
                v.push_str(&format!(
                    "  wire signed [31:0] m{i}_{a} = (f{i}_{a} > m{i}_{p}) ? f{i}_{a} : m{i}_{p};\n",
                    p = a - 1
                ));
            }
            for a in 0..q {
                v.push_str(&format!(
                    "  wire [31:0] g{i}_{a} = (m{i}_{} - f{i}_{a}) >> {EXP_GAP_SHIFT};\n",
                    q - 1
                ));
                v.push_str(&format!(
                    "  wire [{}:0] ad{i}_{a} = (g{i}_{a} > 32'd{last}) ? {bits}'d{last} : g{i}_{a}[{}:0];\n",
                    bits - 1,
                    bits - 1
                ));
                v.push_str(&format!("  wire [15:0] w{i}_{a} = ex(ad{i}_{a});\n"));
            }
            v.push_str(&format!("  wire [31:0] c{i}_0 = {{16'd0, w{i}_0}};\n"));
            for a in 1..q {
                v.push_str(&format!(
                    "  wire [31:0] c{i}_{a} = c{i}_{} + {{16'd0, w{i}_{a}}};\n",
                    a - 1
                ));
            }
            v.push_str(&format!("  wire [31:0] d{i} = xs32(rng[{i}]);\n"));
            v.push_str(&format!(
                "  wire [47:0] p{i} = d{i}[31:16] * c{i}_{};\n",
                q - 1
            ));
            v.push_str(&format!("  wire [31:0] t{i} = p{i}[47:16];\n"));
            // The walk: the first a with t < c_a, from the top down as nested selects.
            let mut sel = format!("{sb}'d{}", q - 1);
            for a in (0..q - 1).rev() {
                sel = format!("(t{i} < c{i}_{a}) ? {sb}'d{a} : ({sel})");
            }
            v.push_str(&format!("  wire [{}:0] k{i} = {sel};\n", sb - 1));
        }
        v.push_str(&format!(
            "\n  always @(posedge clk) begin\n    if (rst) begin\n      phase <= {pb}'d0;\n"
        ));
        for i in 0..n {
            v.push_str(&format!("      rng[{i}] <= 32'd{};\n", b.seeds[i]));
            v.push_str(&format!(
                "      state[{}:{}] <= {sb}'d{};\n",
                i * sb + sb - 1,
                i * sb,
                b.init_s[i]
            ));
        }
        v.push_str(&format!(
            "    end else if (en) begin\n      phase <= (phase == {pb}'d{}) ? {pb}'d0 : phase + {pb}'d1;\n",
            classes - 1
        ));
        for (ci, class) in b.classes.iter().enumerate() {
            v.push_str(&format!("      if (phase == {pb}'d{ci}) begin\n"));
            for &iu in class {
                let i = iu as usize;
                v.push_str(&format!(
                    "        state[{}:{}] <= k{i}; rng[{i}] <= d{i};\n",
                    i * sb + sb - 1,
                    i * sb
                ));
            }
            v.push_str("      end\n");
        }
        v.push_str("    end\n  end\nendmodule\n");
        v
    }

    /// A self-checking testbench and the emulator's packed per-sweep trace, as
    /// [`FixedPdit::emit_testbench`].
    pub fn emit_testbench(&mut self, module: &str, sweeps: usize) -> (String, String) {
        self.reset();
        let mut expected = String::new();
        for _ in 0..sweeps {
            self.sweep();
            expected.push_str(&self.base.packed_hex());
            expected.push('\n');
        }
        self.reset();
        let top = self.base.n * self.base.state_bits() - 1;
        let ptop = self.base.phase_bits() - 1;
        let classes = self.base.classes.len();
        let tb = format!(
            "`timescale 1ns/1ps\nmodule tb;\n  reg clk = 0, rst = 1;\n  wire [{top}:0] state;\n  wire [{ptop}:0] phase;\n  {module} dut(.clk(clk), .rst(rst), .en(1'b1), .state(state), .phase(phase));\n  reg [{top}:0] expected [0:{last}];\n  integer sw, errors = 0;\n  always #5 clk = ~clk;\n  initial begin\n    $readmemh(\"expected.hex\", expected);\n    @(posedge clk); @(posedge clk); @(negedge clk); rst = 0;\n    for (sw = 0; sw < {sweeps}; sw = sw + 1) begin\n      repeat ({classes}) @(posedge clk);\n      #1;\n      if (state !== expected[sw]) begin\n        errors = errors + 1;\n        $display(\"MISMATCH sweep %0d: got %h want %h\", sw, state, expected[sw]);\n      end\n    end\n    if (errors == 0) $display(\"FERROTHERM_PASS\");\n    else $display(\"FERROTHERM_FAIL %0d\", errors);\n    $finish;\n  end\nendmodule\n",
            last = sweeps - 1,
        );
        (tb, expected)
    }
}

/// `index = sum_i s_i q^i`, site 0 least significant, as [`Potts::index_of`].
fn state_of(mut index: usize, q: usize, n: usize) -> Vec<u8> {
    let mut s = vec![0u8; n];
    for v in &mut s {
        *v = (index % q) as u8;
        index /= q;
    }
    s
}

/// The dense kernel of one sweep: the product of single-site kernels in `order`, each site's
/// conditional supplied by `cond(site, state)`.
fn dense_sweep_kernel(
    q: usize,
    n: usize,
    order: &[usize],
    cond: impl Fn(usize, &[u8]) -> Vec<f64>,
) -> Result<Vec<f64>, Unfit> {
    let states = u32::try_from(n)
        .ok()
        .and_then(|e| q.checked_pow(e))
        .filter(|&m| m <= MAX_DENSE_STATES);
    let Some(m) = states else {
        return Err(Unfit::TooManyStates { q, n });
    };
    let stride: Vec<usize> = (0..n).map(|i| q.pow(i as u32)).collect();
    let mut kernel = vec![0.0f64; m * m];
    let mut dist = vec![0.0f64; m];
    let mut next = vec![0.0f64; m];
    for x in 0..m {
        dist.iter_mut().for_each(|v| *v = 0.0);
        dist[x] = 1.0;
        for &i in order {
            next.iter_mut().for_each(|v| *v = 0.0);
            for y in 0..m {
                let p = dist[y];
                if p == 0.0 {
                    continue;
                }
                let s = state_of(y, q, n);
                let law = cond(i, &s);
                let base = y - usize::from(s[i]) * stride[i];
                for (a, &pa) in law.iter().enumerate() {
                    next[base + a * stride[i]] += p * pa;
                }
            }
            core::mem::swap(&mut dist, &mut next);
        }
        kernel[x * m..(x + 1) * m].copy_from_slice(&dist);
    }
    Ok(kernel)
}

/// The stationary law of a dense row-stochastic kernel, by a direct solve of `pi (I - P) = 0`
/// with the last equation replaced by `sum pi = 1`.
///
/// # Errors
///
/// [`Unfit::Reducible`] when the system is singular: more than one closed class.
pub fn stationary_of(kernel: &[f64], m: usize) -> Result<Vec<f64>, Unfit> {
    let mut a = vec![0.0f64; m * m];
    for x in 0..m {
        for y in 0..m {
            let delta = if x == y { 1.0 } else { 0.0 };
            a[y * m + x] = delta - kernel[x * m + y];
        }
    }
    for c in 0..m {
        a[(m - 1) * m + c] = 1.0;
    }
    let mut b = vec![0.0f64; m];
    b[m - 1] = 1.0;
    if !autocorr::lu_solve(&mut a, m, &mut b, 1) {
        return Err(Unfit::Reducible);
    }
    let total: f64 = b.iter().map(|v| v.max(0.0)).sum();
    Ok(b.iter().map(|v| v.max(0.0) / total).collect())
}

/// Kemeny's constant `K = tr Z - 1` of a dense kernel with stationary law `pi`, where
/// `Z = (I - P + 1 pi^T)^-1`: the expected steps from any state to one drawn from `pi`.
///
/// It compares kernels on the SAME state space and nothing else. The expected time to reach a
/// target `y` is at least `1/pi_y` even for a kernel that mixes in one step, and averaging that
/// over `pi` gives `m - 1`: a perfectly mixing chain on `m` states has `K = m - 1`, so `K` counts
/// states before it measures anything. The first draft of this module compared a 27-state native
/// chain to its 512-state one-hot spelling by `K` and read the spelling as six times slower for
/// no better reason than that it has nineteen times the states. [`mixing_time`] and
/// [`tau_int_of`] are the size-free measures [`native_row`] and [`encoded_row`] report.
///
/// # Errors
///
/// [`Unfit::Reducible`] when the fundamental matrix does not exist.
pub fn kemeny_of(kernel: &[f64], m: usize, pi: &[f64]) -> Result<f64, Unfit> {
    let mut a = vec![0.0f64; m * m];
    for x in 0..m {
        for y in 0..m {
            let delta = if x == y { 1.0 } else { 0.0 };
            a[x * m + y] = delta - kernel[x * m + y] + pi[y];
        }
    }
    let mut z = vec![0.0f64; m * m];
    for i in 0..m {
        z[i * m + i] = 1.0;
    }
    if !autocorr::lu_solve(&mut a, m, &mut z, m) {
        return Err(Unfit::Reducible);
    }
    let trace: f64 = (0..m).map(|i| z[i * m + i]).sum();
    Ok(trace - 1.0)
}

/// One line of the native-versus-encoded comparison.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Row {
    /// How the variables were realised.
    pub embodiment: Embodiment,
    /// The penalty an encoding was held at; zero for a native unit.
    pub penalty: f64,
    /// States of the chain.
    pub states: usize,
    /// Noise draws one sweep of the whole model spends: `qn` native or one-hot, `(q-1)n`
    /// domain-wall.
    pub draws_per_sweep: usize,
    /// Sweeps until every start is within total variation `1/4` of the stationary law, or `None`
    /// if [`MIXING_SWEEPS_CAP`] sweeps were not enough.
    pub mixing_sweeps: Option<usize>,
    /// [`Row::mixing_sweeps`] in noise draws.
    pub mixing_draws: Option<usize>,
    /// Integrated autocorrelation time, in sweeps, of the indicator that the first variable holds
    /// its first value -- zero on codes that decode to nothing.
    pub tau_int_sweeps: f64,
    /// [`Row::tau_int_sweeps`] in noise draws.
    pub tau_int_draws: f64,
    /// Stationary mass on states that decode to a value: one for a native unit.
    pub valid_mass: f64,
}

/// The most sweeps [`mixing_time`] iterates before reporting `None`, and the cap the example uses;
/// each sweep past the first is a dense `m^3` product, so a test passes a small cap or zero.
pub const MIXING_SWEEPS_CAP: usize = 2048;

/// Sweeps until the worst start is within total variation `eps` of `pi`: the smallest `t` with
/// `max_x TV(P^t(x, .), pi) <= eps`, by dense powers of the kernel, or `None` past `cap`.
#[must_use]
pub fn mixing_time(kernel: &[f64], m: usize, pi: &[f64], eps: f64, cap: usize) -> Option<usize> {
    let worst = |pt: &[f64]| {
        (0..m)
            .map(|x| autocorr::total_variation(&pt[x * m..(x + 1) * m], pi))
            .fold(0.0f64, f64::max)
    };
    let mut pt = kernel.to_vec();
    let mut next = vec![0.0f64; m * m];
    for t in 1..=cap {
        if worst(&pt) <= eps {
            return Some(t);
        }
        for x in 0..m {
            let row = &pt[x * m..(x + 1) * m];
            let out = &mut next[x * m..(x + 1) * m];
            out.iter_mut().for_each(|v| *v = 0.0);
            for (y, &p) in row.iter().enumerate() {
                if p == 0.0 {
                    continue;
                }
                for (o, &k) in out.iter_mut().zip(&kernel[y * m..(y + 1) * m]) {
                    *o += p * k;
                }
            }
        }
        core::mem::swap(&mut pt, &mut next);
    }
    None
}

/// The integrated autocorrelation time of an observable `f` under a dense kernel with stationary
/// law `pi`, in kernel steps: `2 (f~ D Z f~) / (f~ D f~) - 1` with `f~ = f - <f>`, `D = diag pi`
/// and `Z` the fundamental matrix, which sums every lag exactly.
///
/// # Errors
///
/// [`Unfit::Reducible`] when the fundamental matrix does not exist, and for an observable with no
/// variance under `pi`, which has no autocorrelation time.
pub fn tau_int_of(kernel: &[f64], m: usize, pi: &[f64], f: &[f64]) -> Result<f64, Unfit> {
    let mean: f64 = pi.iter().zip(f).map(|(p, v)| p * v).sum();
    let centred: Vec<f64> = f.iter().map(|v| v - mean).collect();
    let var: f64 = pi.iter().zip(&centred).map(|(p, v)| p * v * v).sum();
    if var <= 0.0 {
        return Err(Unfit::Reducible);
    }
    let mut a = vec![0.0f64; m * m];
    for x in 0..m {
        for y in 0..m {
            let delta = if x == y { 1.0 } else { 0.0 };
            a[x * m + y] = delta - kernel[x * m + y] + pi[y];
        }
    }
    let mut zf = centred.clone();
    if !autocorr::lu_solve(&mut a, m, &mut zf, 1) {
        return Err(Unfit::Reducible);
    }
    let sum: f64 = pi
        .iter()
        .zip(&centred)
        .zip(&zf)
        .map(|((p, c), z)| p * c * z)
        .sum();
    Ok(2.0 * sum / var - 1.0)
}

/// The native chain: exact heat-bath updates of `m`'s sites in index order, its stationary law
/// checked against [`crate::potts::enumerate`] by the tests rather than assumed. `mixing_cap` is
/// the most sweeps the mixing time is followed for; zero skips it and reports `None`.
///
/// # Errors
///
/// [`Unfit::TooManyStates`] above [`MAX_DENSE_STATES`]; [`Unfit::Reducible`] if the chain is.
pub fn native_row(m: &Potts, beta: f64, mixing_cap: usize) -> Result<Row, Unfit> {
    let (q, n) = (m.q(), m.n());
    let adj = adjacency(m);
    let order: Vec<usize> = (0..n).collect();
    let kernel = dense_sweep_kernel(q, n, &order, |i, s| {
        let mut fields: Vec<f64> = (0..q).map(|a| m.field(i, a as u8)).collect();
        for &(j, w) in &adj[i] {
            for (a, f) in fields.iter_mut().enumerate() {
                *f += w * m.kind().pair(q, a as u8, s[j]);
            }
        }
        softmax(&fields, beta)
    })?;
    let states = kernel.len() / q.pow(n as u32);
    let pi = stationary_of(&kernel, states)?;
    let first: Vec<f64> = (0..states)
        .map(|x| f64::from(u8::from(state_of(x, q, n)[0] == 0)))
        .collect();
    let tau = tau_int_of(&kernel, states, &pi, &first)?;
    let mixing = mixing_time(&kernel, states, &pi, 0.25, mixing_cap);
    let draws_per_sweep = n * q;
    Ok(Row {
        embodiment: Embodiment::GumbelMax,
        penalty: 0.0,
        states,
        draws_per_sweep,
        mixing_sweeps: mixing,
        mixing_draws: mixing.map(|t| t * draws_per_sweep),
        tau_int_sweeps: tau,
        tau_int_draws: tau * draws_per_sweep as f64,
        valid_mass: 1.0,
    })
}

/// The indicator `[value = a]` of a `q`-valued variable as an affine function of its spins:
/// `(constant, [(local spin, coefficient)])`.
fn indicator(encoding: Encoding, q: usize, a: usize) -> (f64, Vec<(usize, f64)>) {
    match encoding {
        Encoding::OneHot => (0.5, vec![(a, 0.5)]),
        Encoding::DomainWall => {
            // value v: the first v spins are +1, the rest -1; [v = a] = (s_{a-1} - s_a) / 2 with
            // s_{-1} = +1 and s_{q-1} = -1 fixed.
            let mut terms = Vec::new();
            let mut c = 0.0;
            if a == 0 {
                c += 0.5;
            } else {
                terms.push((a - 1, 0.5));
            }
            if a == q - 1 {
                c += 0.5;
            } else {
                terms.push((a, -0.5));
            }
            (c, terms)
        }
        Encoding::Binary => unreachable!("binary is not an embodiment here"),
    }
}

/// Lower a Potts model to spins under `encoding` with a penalty of `penalty` per variable:
/// the graph, one slot per site, and the constant `c` with
/// `E_potts(s) = E_ising(spins(s)) + c` on every valid codeword when `penalty = 0`.
///
/// # Panics
///
/// If `encoding` is binary, which is not exact for `q` not a power of two and not an embodiment
/// this module compares.
#[must_use]
pub fn lower(m: &Potts, encoding: Encoding, penalty: f64) -> (Graph, Vec<Slot>, f64) {
    assert!(
        encoding != Encoding::Binary,
        "binary is not an embodiment this module compares"
    );
    let (q, n) = (m.q(), m.n());
    let width = encoding.spins(q);
    let slots: Vec<Slot> = (0..n).map(|i| Slot::new(i * width, q, encoding)).collect();
    let mut b = GraphBuilder::new(n * width);
    let mut constant = 0.0;
    for (i, j, w) in m.edges() {
        for a in 0..q {
            let (ci, ti) = indicator(encoding, q, a);
            let (cj, tj) = indicator(encoding, q, a);
            // -w x_ia x_ja, expanded: E_ising carries -J s s - h s, so each product's coefficient
            // lands on the coupling or field with its sign intact.
            constant -= w * ci * cj;
            for &(k, ck) in &ti {
                b.bias(slots[i].base + k, w * ck * cj);
                for &(l, dl) in &tj {
                    b.couple(slots[i].base + k, slots[j].base + l, w * ck * dl);
                }
            }
            for &(l, dl) in &tj {
                b.bias(slots[j].base + l, w * ci * dl);
            }
        }
    }
    for i in 0..n {
        for a in 0..q {
            let h = m.field(i, a as u8);
            if h == 0.0 {
                continue;
            }
            let (c, terms) = indicator(encoding, q, a);
            constant -= h * c;
            for &(k, ck) in &terms {
                b.bias(slots[i].base + k, h * ck);
            }
        }
    }
    if penalty != 0.0 {
        for slot in &slots {
            let exact = slot.add_penalty(&mut b, penalty);
            debug_assert!(exact, "one-hot and domain-wall penalties are exact");
        }
    }
    (b.build(), slots, constant)
}

/// The most spins an encoded chain is built densely over: `2^10` states.
pub const MAX_ENCODED_SPINS: usize = 10;

/// The encoded chain: one sequential Gibbs sweep over the spins of [`lower`] as a dense kernel,
/// its Boltzmann law, the mixing time (followed for at most `mixing_cap` sweeps, zero to skip)
/// and the autocorrelation time of the same indicator [`native_row`] uses (zero on codes that
/// decode to nothing), and the mass on decodable states.
///
/// # Errors
///
/// [`Unfit::TooManyStates`] above [`MAX_ENCODED_SPINS`] spins; [`Unfit::Reducible`] as the
/// solves report it.
///
/// # Panics
///
/// If `embodiment` is not an encoded one.
pub fn encoded_row(
    m: &Potts,
    beta: f64,
    embodiment: Embodiment,
    penalty: f64,
    mixing_cap: usize,
) -> Result<Row, Unfit> {
    let encoding = embodiment.encoding().expect("an encoded embodiment");
    let (g, slots, _) = lower(m, encoding, penalty);
    if g.n > MAX_ENCODED_SPINS {
        return Err(Unfit::TooManyStates { q: 2, n: g.n });
    }
    let states = 1usize << g.n;
    let mut kernel = vec![0.0f64; states * states];
    let mut point = vec![0.0f64; states];
    for x in 0..states {
        point[x] = 1.0;
        let row = autocorr::apply_distribution(&g, beta, Kernel::SequentialGibbs, &point);
        point[x] = 0.0;
        kernel[x * states..(x + 1) * states].copy_from_slice(&row);
    }
    let pi = stationary_of(&kernel, states)?;
    let mut valid_mass = 0.0;
    let mut first = vec![0.0f64; states];
    for x in 0..states {
        let spins = autocorr::spins(x, g.n);
        if slots.iter().all(|slot| slot.decode(&spins).is_some()) {
            valid_mass += pi[x];
        }
        if slots[0].decode(&spins) == Some(0) {
            first[x] = 1.0;
        }
    }
    let tau = tau_int_of(&kernel, states, &pi, &first)?;
    let mixing = mixing_time(&kernel, states, &pi, 0.25, mixing_cap);
    let draws_per_sweep = g.n;
    Ok(Row {
        embodiment,
        penalty,
        states,
        draws_per_sweep,
        mixing_sweeps: mixing,
        mixing_draws: mixing.map(|t| t * draws_per_sweep),
        tau_int_sweeps: tau,
        tau_int_draws: tau * draws_per_sweep as f64,
        valid_mass,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::potts::{enumerate, ring, PottsBuilder};

    /// `softmax([f, -f])` is `sigma(2 beta f)`: the p-bit is the two-state p-dit, to `1e-15`. The
    /// radix economy is minimised at three among the integers, near `e` among the reals, and the
    /// three charges give the three verdicts the module documentation tabulates.
    #[test]
    fn the_p_bit_is_the_two_state_p_dit_and_the_radix_economy_is_hartleys() {
        for &beta in &[0.5f64, 1.0, 2.0] {
            for &f in &[-2.0f64, -0.3, 0.0, 0.7, 3.0] {
                let law = softmax(&[f, -f], beta);
                let want = crate::kernel::p_up(f, beta);
                assert!(
                    (law[0] - want).abs() < 1e-15,
                    "f {f} beta {beta}: {} vs {want}",
                    law[0]
                );
            }
        }
        assert_eq!(best_integer_radix(), 3);
        assert!(radix_economy(3.0) < radix_economy(2.0));
        assert!(
            radix_economy(2.0) < radix_economy(4.0) + 1e-12
                && radix_economy(4.0) < radix_economy(2.0) + 1e-12
        );
        assert!(radix_economy(core::f64::consts::E) < radix_economy(3.0));
        assert!(radix_economy(core::f64::consts::E) < radix_economy(2.5));
        let ternary = Embodiment::GumbelMax;
        let binary = Embodiment::Comparator;
        assert!(
            bits_per_charge(3, ternary, Charge::Radix) > bits_per_charge(2, binary, Charge::Radix)
        );
        assert!(
            bits_per_charge(3, ternary, Charge::Comparators)
                < bits_per_charge(2, binary, Charge::Comparators)
        );
        assert!(
            bits_per_charge(3, ternary, Charge::Draws) < bits_per_charge(2, binary, Charge::Draws)
        );
        assert_eq!(
            Embodiment::OneHot.cost(3).draws,
            Embodiment::GumbelMax.cost(3).draws
        );
        assert!((wasted_codes(Encoding::OneHot, 3) - 5.0 / 8.0).abs() < 1e-15);
        assert!((wasted_codes(Encoding::DomainWall, 3) - 1.0 / 4.0).abs() < 1e-15);
        assert!(!Embodiment::Comparator.supports(3));
    }

    /// The ROM unit's exact law sums to one, is the softmax to a few parts in a thousand where
    /// the fields are close, and is exactly zero for a state whose field trails by more than the
    /// ROM's span -- the floor, which more bits push out.
    #[test]
    fn the_rom_units_exact_law_is_the_softmax_until_the_gumbel_range_ends() {
        let scale = f64::from(1u32 << FRAC);
        for &bits in &[10u32, 14] {
            let rom = gumbel_rom(bits);
            let fields = [0.0f64, -1.0, -2.0];
            let fields_q: Vec<i32> = fields.iter().map(|f| (f * scale).round() as i32).collect();
            let law = rom_conditional(&fields_q, &rom);
            assert!((law.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            let tv = autocorr::total_variation(&law, &softmax(&fields, 1.0));
            let allowed = if bits == 10 { 0.02 } else { 0.005 };
            assert!(tv < allowed, "{bits} bits: TV {tv} from the softmax");
        }
        let rom = gumbel_rom(10);
        let span = rom_span(&rom);
        assert!(span > 9.0 && span < 10.5, "ten-bit span {span}");
        let beyond = [0i32, -((span * scale) as i32) - 1, 0];
        let law = rom_conditional(&beyond, &rom);
        assert_eq!(law[1], 0.0, "beyond the span the state cannot win: {law:?}");
        let within = [0i32, -((span * scale) as i32) + 8, 0];
        assert!(rom_conditional(&within, &rom)[1] > 0.0);
        assert!(rom_span(&gumbel_rom(14)) > span, "more bits, wider span");
    }

    /// The fixed-point p-trit fabric on a three-state ring: its exact sweep kernel's stationary
    /// law is the Potts law to within the ROM's quantisation, and a run of the emulator lands on
    /// that law (counts, so valid on a busy machine).
    #[test]
    fn the_fixed_point_p_trit_fabric_samples_the_potts_law() {
        let m = ring(4, 3, 1.0, Interaction::Potts);
        let beta = 0.6;
        let exact = enumerate(&m, beta).expect("81 states");
        let mut fab = FixedPdit::new(&m, beta, 7, 12).expect("a Potts ring fits");
        let kernel = fab.sweep_kernel().expect("81 states");
        let pi = stationary_of(&kernel, 81).expect("irreducible");
        let tv = autocorr::total_variation(&pi, &exact.p);
        assert!(tv < 0.01, "fabric law vs Potts law: TV {tv}");
        let mut counts = vec![0.0f64; 81];
        let sweeps = 40_000;
        for _ in 0..sweeps {
            fab.sweep();
            counts[m.index_of(&fab.s).expect("a state the model reads")] += 1.0;
        }
        let hist: Vec<f64> = counts.iter().map(|c| c / f64::from(sweeps)).collect();
        let tv = autocorr::total_variation(&hist, &pi);
        assert!(
            tv < 0.03,
            "emulator histogram vs its own exact law: TV {tv}"
        );
        assert_eq!(fab.classes.len(), 2, "a four-ring is bipartite");
    }

    /// Both lowerings agree with the Potts energy on every valid codeword, up to the returned
    /// constant with no penalty and up to a constant with one, on a ring with fields.
    #[test]
    fn the_lowerings_agree_with_the_potts_energy_on_every_valid_codeword() {
        let mut b = PottsBuilder::new(3, 3, Interaction::Potts);
        b.couple(0, 1, 1.0);
        b.couple(1, 2, 0.5);
        b.couple(2, 0, -0.7);
        b.field(0, 1, 0.3);
        b.field(2, 0, -0.4);
        let m = b.build();
        for encoding in [Encoding::OneHot, Encoding::DomainWall] {
            for penalty in [0.0f64, 2.0] {
                let (g, slots, constant) = lower(&m, encoding, penalty);
                let mut offset = None;
                for index in 0..27usize {
                    let s = state_of(index, 3, 3);
                    let mut spins = vec![0i8; g.n];
                    for (slot, &v) in slots.iter().zip(&s) {
                        slot.encode(usize::from(v), &mut spins);
                    }
                    let diff = g.energy(&spins) + constant - m.energy(&s).expect("valid");
                    if penalty == 0.0 {
                        assert!(diff.abs() < 1e-12, "{encoding:?} state {s:?}: {diff}");
                    } else {
                        let o = *offset.get_or_insert(diff);
                        assert!(
                            (diff - o).abs() < 1e-12,
                            "{encoding:?} state {s:?}: offset {diff} vs {o}"
                        );
                    }
                }
            }
        }
    }

    /// The native chain's stationary law is the enumeration's; Kemeny's constant on it is the
    /// state count less one, which is why the rows do not report it; and on a three-state triangle
    /// at `beta = 0.5` the encodings leave mass on codes that decode to nothing while the native
    /// chain mixes in one sweep.
    #[test]
    fn a_native_p_trit_reaches_stationarity_in_fewer_draws_than_its_encodings() {
        let m = ring(3, 3, 1.0, Interaction::Potts);
        let beta = 0.5;
        let native = native_row(&m, beta, 8).expect("27 states");
        let adj = adjacency(&m);
        let order: Vec<usize> = (0..3).collect();
        let kernel = dense_sweep_kernel(3, 3, &order, |i, s| {
            let mut fields = vec![0.0f64; 3];
            for &(j, w) in &adj[i] {
                fields[usize::from(s[j])] += w;
            }
            softmax(&fields, beta)
        })
        .expect("27 states");
        let pi = stationary_of(&kernel, 27).expect("irreducible");
        let exact = enumerate(&m, beta).expect("27 states");
        assert!(autocorr::total_variation(&pi, &exact.p) < 1e-10);
        let k = kemeny_of(&kernel, 27, &pi).expect("irreducible");
        assert!(
            (k - 26.0).abs() < 1.0,
            "Kemeny counts states: {k} on 27 states"
        );
        assert_eq!(native.mixing_sweeps, Some(1), "{native:?}");
        for embodiment in [Embodiment::OneHot, Embodiment::DomainWall] {
            for penalty in [1.0f64, 2.0, 4.0] {
                let row = encoded_row(&m, beta, embodiment, penalty, 0).expect("dense");
                assert!(row.valid_mass < 1.0 && row.valid_mass > 0.0, "{row:?}");
                assert!(
                    row.mixing_sweeps.is_none(),
                    "cap zero skips the powers: {row:?}"
                );
                assert!(row.tau_int_sweeps > 0.0, "{row:?}");
                println!(
                    "{} at {penalty}: mixing {:?} sweeps ({:?} draws), tau_int {:.3} sweeps ({:.2} draws), valid {:.4} -- native {:?} sweeps ({:?} draws), tau_int {:.3} ({:.2} draws)",
                    embodiment.label(),
                    row.mixing_sweeps,
                    row.mixing_draws,
                    row.tau_int_sweeps,
                    row.tau_int_draws,
                    row.valid_mass,
                    native.mixing_sweeps,
                    native.mixing_draws,
                    native.tau_int_sweeps,
                    native.tau_int_draws
                );
            }
        }
    }

    /// The verdict, cold: on the three-state triangle at `beta = 2`, at the smallest penalty on
    /// the grid `0.5, 1, 2, 4` that puts 99% of the stationary mass on decodable states, the
    /// domain-wall spelling's autocorrelation time in draws is within a factor of two of the
    /// native p-trit's, and the one-hot spelling's is more than a hundred times it. The penalty
    /// that makes an encoding valid is the penalty that freezes it, and one-hot needs four times
    /// the domain wall's.
    #[test]
    fn the_penalty_that_makes_one_hot_valid_freezes_it_and_the_native_unit_needs_none() {
        let m = ring(3, 3, 1.0, Interaction::Potts);
        let beta = 2.0;
        let native = native_row(&m, beta, 0).expect("27 states");
        let mut at_99 = Vec::new();
        for embodiment in [Embodiment::DomainWall, Embodiment::OneHot] {
            let mut chosen = None;
            for penalty in [0.5f64, 1.0, 2.0, 4.0] {
                let row = encoded_row(&m, beta, embodiment, penalty, 0).expect("dense");
                if row.valid_mass >= 0.99 {
                    chosen = Some(row);
                    break;
                }
            }
            at_99.push(chosen.expect("some penalty on the grid reaches 99% valid"));
        }
        let (wall, hot) = (at_99[0], at_99[1]);
        assert!(
            (wall.penalty - 1.0).abs() < 1e-12 && (hot.penalty - 4.0).abs() < 1e-12,
            "{wall:?} {hot:?}"
        );
        assert!(
            wall.tau_int_draws < 2.0 * native.tau_int_draws,
            "domain wall {} vs native {} draws",
            wall.tau_int_draws,
            native.tau_int_draws
        );
        assert!(
            hot.tau_int_draws > 100.0 * native.tau_int_draws,
            "one-hot {} vs native {} draws",
            hot.tau_int_draws,
            native.tau_int_draws
        );
    }

    /// The cumulative unit's exact law sums to one, is the softmax to a few parts in a thousand,
    /// and is exactly zero for a state whose gap exceeds the sixteen-bit weight floor `11.78` --
    /// wider ROMs do not move it, because the floor is the weight's, not the address's.
    #[test]
    fn the_cumulative_units_exact_law_is_the_softmax_until_the_weight_width_ends() {
        let scale = f64::from(1u32 << FRAC);
        let floor = cumulative_floor();
        assert!((floor - 11.78).abs() < 0.01, "{floor}");
        for &bits in &[10u32, 14] {
            let rom = exp_rom(bits);
            let fields = [0.0f64, -1.0, -2.0];
            let fields_q: Vec<i32> = fields.iter().map(|f| (f * scale).round() as i32).collect();
            let law = cumulative_conditional(&fields_q, &rom);
            assert!((law.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            let tv = autocorr::total_variation(&law, &softmax(&fields, 1.0));
            assert!(tv < 5e-3, "{bits} bits: TV {tv} from the softmax");
            let beyond = [0i32, -((floor * scale) as i32) - 8, 0];
            assert_eq!(cumulative_conditional(&beyond, &rom)[1], 0.0);
            let within = [0i32, -((floor * scale) as i32) + 64, 0];
            assert!(cumulative_conditional(&within, &rom)[1] > 0.0);
        }
    }

    /// The cumulative fabric on the three-state four-ring: its exact sweep kernel's stationary
    /// law is the Potts law to within the quantisation, and a run of the emulator lands on it.
    #[test]
    fn the_cumulative_p_trit_fabric_samples_the_potts_law() {
        let m = ring(4, 3, 1.0, Interaction::Potts);
        let beta = 0.6;
        let exact = enumerate(&m, beta).expect("81 states");
        let mut fab = FixedCumulative::new(&m, beta, 11, 10).expect("a Potts ring fits");
        let kernel = fab.sweep_kernel().expect("81 states");
        let pi = stationary_of(&kernel, 81).expect("irreducible");
        let tv = autocorr::total_variation(&pi, &exact.p);
        assert!(tv < 0.01, "cumulative fabric law vs Potts law: TV {tv}");
        let mut counts = vec![0.0f64; 81];
        let sweeps = 40_000;
        for _ in 0..sweeps {
            fab.sweep();
            counts[m.index_of(&fab.base.s).expect("a state the model reads")] += 1.0;
        }
        let hist: Vec<f64> = counts.iter().map(|c| c / f64::from(sweeps)).collect();
        let tv = autocorr::total_variation(&hist, &pi);
        assert!(
            tv < 0.03,
            "emulator histogram vs its own exact law: TV {tv}"
        );
    }

    /// THE HARDWARE GATE for the cumulative p-dit: the same two models as the Gumbel-max gate,
    /// replayed bit-exactly under icarus-verilog, multiply and walk included.
    #[test]
    fn the_cumulative_p_trit_rtl_matches_the_emulator_bit_exact() {
        if std::process::Command::new("iverilog")
            .arg("-V")
            .output()
            .is_err()
        {
            eprintln!("SKIP: iverilog not installed; the cumulative p-dit RTL gate did not run");
            return;
        }
        let mut b = PottsBuilder::new(3, 6, Interaction::Potts);
        for i in 0..6 {
            b.couple(i, (i + 1) % 6, 1.0);
        }
        b.field(0, 2, 0.4);
        b.field(3, 0, -0.6);
        let ring6 = b.build();
        let mut b = PottsBuilder::new(5, 4, Interaction::Potts);
        b.couple(0, 1, 0.8);
        b.couple(1, 2, -0.5);
        b.couple(2, 0, 0.3);
        b.couple(2, 3, 1.1);
        b.field(3, 4, 0.5);
        let five = b.build();
        for (name, m, beta, bits) in [("ring6", &ring6, 0.7, 10u32), ("five", &five, 0.9, 12)] {
            let mut fab = FixedCumulative::new(m, beta, 0x5EED, bits).expect("Potts");
            let rtl = fab.emit_verilog("cpdit");
            let (tb, expected) = fab.emit_testbench("cpdit", 40);
            let dir = std::env::temp_dir()
                .join(format!("ferrotherm_cpdit_{name}_{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("cpdit.v"), rtl).unwrap();
            std::fs::write(dir.join("tb.v"), tb).unwrap();
            std::fs::write(dir.join("expected.hex"), expected).unwrap();
            let out = std::process::Command::new("iverilog")
                .current_dir(&dir)
                .args(["-g2012", "-o", "sim", "cpdit.v", "tb.v"])
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "iverilog ({name}): {}",
                String::from_utf8_lossy(&out.stderr)
            );
            let run = std::process::Command::new("vvp")
                .current_dir(&dir)
                .arg("sim")
                .output()
                .unwrap();
            let stdout = String::from_utf8_lossy(&run.stdout);
            assert!(
                stdout.contains("FERROTHERM_PASS"),
                "cumulative p-dit RTL/emulator divergence ({name}):\n{stdout}"
            );
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// THE HARDWARE GATE for the p-dit: the emitted Verilog, simulated with icarus-verilog, must
    /// reproduce the emulator's packed state trace bit-exactly for every sweep -- on a three-state
    /// ring with fields, two bits per site and two colour classes, and on a five-state four-site
    /// model with a triangle in it: three bits per site and three colour classes. Skips with a
    /// notice if iverilog is not installed.
    #[test]
    fn the_p_trit_rtl_matches_the_emulator_bit_exact() {
        if std::process::Command::new("iverilog")
            .arg("-V")
            .output()
            .is_err()
        {
            eprintln!("SKIP: iverilog not installed; the p-dit RTL bit-exactness gate did not run");
            return;
        }
        let mut b = PottsBuilder::new(3, 6, Interaction::Potts);
        for i in 0..6 {
            b.couple(i, (i + 1) % 6, 1.0);
        }
        b.field(0, 2, 0.4);
        b.field(3, 0, -0.6);
        let ring6 = b.build();
        let mut b = PottsBuilder::new(5, 4, Interaction::Potts);
        b.couple(0, 1, 0.8);
        b.couple(1, 2, -0.5);
        b.couple(2, 0, 0.3);
        b.couple(2, 3, 1.1);
        b.field(3, 4, 0.5);
        let five = b.build();
        for (name, m, beta, bits) in [("ring6", &ring6, 0.7, 10u32), ("five", &five, 0.9, 12)] {
            let mut fab = FixedPdit::new(m, beta, 0x5EED, bits).expect("Potts");
            let rtl = fab.emit_verilog("pdit");
            let (tb, expected) = fab.emit_testbench("pdit", 40);
            let dir =
                std::env::temp_dir().join(format!("ferrotherm_pdit_{name}_{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("pdit.v"), rtl).unwrap();
            std::fs::write(dir.join("tb.v"), tb).unwrap();
            std::fs::write(dir.join("expected.hex"), expected).unwrap();
            let out = std::process::Command::new("iverilog")
                .current_dir(&dir)
                .args(["-g2012", "-o", "sim", "pdit.v", "tb.v"])
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "iverilog ({name}): {}",
                String::from_utf8_lossy(&out.stderr)
            );
            let run = std::process::Command::new("vvp")
                .current_dir(&dir)
                .arg("sim")
                .output()
                .unwrap();
            let stdout = String::from_utf8_lossy(&run.stdout);
            assert!(
                stdout.contains("FERROTHERM_PASS"),
                "p-dit RTL/emulator divergence ({name}):\n{stdout}"
            );
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}
