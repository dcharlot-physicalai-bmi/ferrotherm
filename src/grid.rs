//! Open-grid Gibbs, with each large-grid lever behind its own flag.
//!
//! [`crate::gibbs::Sampler`] walks a CSR. On a large open Ising grid that walk is the wrong
//! shape: every site has the same four-neighbour stencil, the two colours never touch, and the
//! acceptance probability takes nine values. This module is that stencil. It is not a second
//! update rule. [`accept_probability`] is [`crate::kernel::p_up`] of `J` times the neighbour-spin
//! sum, which is the local field.
//!
//! # Levers
//!
//! Each flag is independent. All off is one thread, in row order, drawing one `f64` per site.
//!
//! * `parallel` — the two checkerboard colours, each colour's rows split across threads.
//! * `simd` — spins packed one bit per site, 64 sites of one colour to a word. The field is a
//!   bit-sliced sum of neighbour trits. An even side is required; an odd side is an error, not a
//!   silent scalar sweep.
//! * `tiled` — rows visited in bands of [`TILE_ROWS`], so a colour's halo stays in cache.
//! * `sparse_rng` — the nine acceptance probabilities are a threshold table, and one 64-bit word
//!   feeds eight sites (eight bit-planes) or sixty-four packed sites (one bit-plane). Comparison
//!   starts at the high bit and stops when every site has resolved. The probability is the table
//!   entry over `2^32`, which differs from [`crate::kernel::p_up`] by at most `2^-32`.
//!
//! `parallel` and `tiled` do not change the draws. Each row has its own stream, keyed by
//! `(seed, sweep, colour, row)`, so a different thread count is the same sample. `simd` and
//! `sparse_rng` draw from that same key through a different encoding, so they are the same
//! distribution and not the same bits.
//!
//! The GPU lever is not in this crate. `ferrotherm` stays dependency-free; `ferrotherm-gpu` runs
//! the same chromatic sweep on wgpu. A caller that wants that path has to say so. This module will
//! not substitute it.
//!
//! # Profile
//!
//! [`OpenIsing::set_profile`] times three phases of the sweep that actually runs: reading
//! neighbour spins (`memory_ns`), turning those spins into a field or a threshold (`field_ns`),
//! and drawing the accept bits (`rng_ns`). With `parallel` on, those three are CPU time summed
//! across threads. `wall_ns` is the elapsed time of the call. They are not required to add, and
//! neither number is a joule.

use crate::kernel::p_up;
use crate::rng::Pcg;

/// Rows per cache band. The halo of a band is two extra rows of the other colour.
pub const TILE_ROWS: usize = 32;

/// Sites that share one 64-bit word on the unpacked sparse-rng path.
pub const RNG_GROUP: usize = 8;

/// Lanes in one packed word.
pub const SIMD_LANES: usize = 64;

/// What is on. Construct it by setting fields; there is no default that quietly enables one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Levers {
    /// Checkerboard rows across threads.
    pub parallel: bool,
    /// Bit-packed spins, 64 sites per word.
    pub simd: bool,
    /// Row bands of [`TILE_ROWS`].
    pub tiled: bool,
    /// Threshold table and early-exit bit draws.
    pub sparse_rng: bool,
    /// Neighbour sums from a bitwise popcount, or a kept i8 field, instead of a fresh gather
    /// plus an exponential. With `simd`, the field is the popcount of the four neighbour bit
    /// planes. Without it, each colour keeps an i8 neighbour sum and the next colour reads it.
    pub incremental: bool,
}

impl Levers {
    /// Every lever off. One thread, one `f64` draw per site.
    pub const SCALAR: Self = Self {
        parallel: false,
        simd: false,
        tiled: false,
        sparse_rng: false,
        incremental: false,
    };

    /// Checkerboard rows across threads, bit-packed spins, cache tiles, and a popcount field.
    pub const COMBINED: Self = Self {
        parallel: true,
        simd: true,
        tiled: true,
        sparse_rng: false,
        incremental: true,
    };

    /// One lever by name. Anything else is an error naming the token that arrived.
    ///
    /// # Errors
    ///
    /// The name is not `parallel`, `simd`, `tiled`, `sparse-rng`, `incremental`, or `combined`.
    /// `gpu` is not a lever on this struct. `combined` is the four of parallel, simd, tiled, and
    /// incremental together, not a silent mix with `sparse-rng`.
    pub fn one(name: &str) -> Result<Self, String> {
        let mut levers = Self::SCALAR;
        match name {
            "parallel" => levers.parallel = true,
            "simd" => levers.simd = true,
            "tiled" => levers.tiled = true,
            "sparse-rng" => levers.sparse_rng = true,
            "incremental" => levers.incremental = true,
            "combined" => return Ok(Self::COMBINED),
            other => {
                return Err(format!(
                    "unknown grid lever {other}. The names are parallel, simd, tiled, sparse-rng, \
                     incremental, combined. gpu is a separate path and is not a flag on this struct"
                ));
            }
        }
        Ok(levers)
    }
}

/// Why an open grid could not be built. The message names the value that was refused.
#[derive(Clone, Debug, PartialEq)]
pub struct GridError {
    /// What was wrong, including the value that arrived.
    pub message: String,
}

impl core::fmt::Display for GridError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.message)
    }
}

impl core::error::Error for GridError {}

/// Time and draw counts for one or more sweeps.
///
/// `rng_ns`, `field_ns` and `memory_ns` are zero unless [`OpenIsing::set_profile`] is on. With
/// more than one thread they are summed CPU time, not elapsed time. `wall_ns` is elapsed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SweepProfile {
    /// Sweeps this profile covers.
    pub sweeps: u64,
    /// Elapsed time of those sweeps.
    pub wall_ns: u128,
    /// Time spent drawing accept bits. Summed across threads.
    pub rng_ns: u128,
    /// Time spent turning neighbour spins into a field or a threshold. Summed across threads.
    pub field_ns: u128,
    /// Time spent reading or writing spins. Summed across threads.
    pub memory_ns: u128,
    /// 64-bit random words consumed. A scalar site uses one word for its `f64` draw.
    pub rng_words: u64,
    /// Threads that ran the last sweep. One when `parallel` is off, even if more were offered.
    pub threads_used: usize,
}

impl SweepProfile {
    fn add(&mut self, other: &Self) {
        self.rng_ns += other.rng_ns;
        self.field_ns += other.field_ns;
        self.memory_ns += other.memory_ns;
        self.rng_words += other.rng_words;
    }
}

/// One open `side × side` Ising grid, `E = −J Σ s_i s_j`, no field.
pub struct OpenIsing {
    side: usize,
    j: f64,
    beta: f64,
    seed: u64,
    sweep_idx: u64,
    levers: Levers,
    threads: usize,
    threads_used: usize,
    /// Row-major `+1`/`−1`. Source of truth when `simd` is off. Unpacked from `bits` on read
    /// when a packed sweep has run.
    spins: Vec<i8>,
    /// `bits[colour][row]` packs that colour's sites in the row. Empty unless `simd`.
    bits: Vec<Vec<Vec<u64>>>,
    packed_dirty: bool,
    thresh: [u32; 9],
    /// Neighbour-spin sums kept for the i8 incremental path. Empty when that path is off.
    fields: Vec<i8>,
    profile: bool,
}

impl OpenIsing {
    /// All spins `−1`. `threads` is the count `parallel` may use; it is an error if it is zero.
    ///
    /// # Errors
    ///
    /// A zero side, a non-finite `J` or `beta`, a negative `beta`, zero threads, or `simd` on an
    /// odd side.
    pub fn try_new(
        side: usize,
        j: f64,
        beta: f64,
        seed: u64,
        levers: Levers,
        threads: usize,
    ) -> Result<Self, GridError> {
        if side == 0 {
            return Err(GridError {
                message: "open grid side is 0".into(),
            });
        }
        if !j.is_finite() {
            return Err(GridError {
                message: format!("open grid J is not finite: {j}"),
            });
        }
        if !beta.is_finite() || beta < 0.0 {
            return Err(GridError {
                message: format!("open grid beta must be finite and >= 0, got {beta}"),
            });
        }
        if threads == 0 {
            return Err(GridError {
                message: "open grid threads is 0; pass at least 1".into(),
            });
        }
        if levers.simd && !side.is_multiple_of(2) {
            return Err(GridError {
                message: format!(
                    "simd bit-packing needs an even side, got {side}. It is not run as a scalar sweep"
                ),
            });
        }
        let n = side * side;
        let spins = vec![-1i8; n];
        let bits = if levers.simd {
            pack_all(&spins, side)
        } else {
            Vec::new()
        };
        let mut fields = Vec::new();
        if levers.incremental && !levers.simd {
            fields = vec![0i8; n];
            for y in 0..side {
                for x in 0..side {
                    fields[y * side + x] = neighbor_sum(&spins, side, x, y) as i8;
                }
            }
        }
        Ok(Self {
            side,
            j,
            beta,
            seed,
            sweep_idx: 0,
            levers,
            threads,
            threads_used: 1,
            spins,
            bits,
            packed_dirty: false,
            thresh: threshold_table(j, beta),
            fields,
            profile: false,
        })
    }

    /// Record per-phase times. Off by default, so a cost measurement is not paying for clocks.
    pub fn set_profile(&mut self, on: bool) {
        self.profile = on;
    }

    /// The levers this grid was built with.
    #[must_use]
    pub fn levers(&self) -> Levers {
        self.levers
    }

    /// Side length.
    #[must_use]
    pub fn side(&self) -> usize {
        self.side
    }

    /// Threads the last sweep actually spawned.
    #[must_use]
    pub fn threads_used(&self) -> usize {
        self.threads_used
    }

    /// Spins in row-major order, `+1` or `−1`.
    #[must_use]
    pub fn spins(&mut self) -> &[i8] {
        self.sync_spins();
        &self.spins
    }

    /// `E = −J Σ_{⟨ij⟩} s_i s_j` over the open grid. The same sum as [`crate::graph::Graph::energy`].
    #[must_use]
    pub fn energy(&mut self) -> f64 {
        self.sync_spins();
        grid_energy(&self.spins, self.side, self.j)
    }

    /// One chromatic sweep. Colour `(x + y) % 2 == 0` first, then the other colour.
    pub fn sweep(&mut self) -> SweepProfile {
        let wall = std::time::Instant::now();
        let sweep = self.sweep_idx;
        let mut prof = SweepProfile::default();
        let threads = if self.levers.parallel {
            self.threads.min(self.side).max(1)
        } else {
            1
        };
        self.threads_used = threads;
        if self.levers.simd {
            self.sweep_simd(sweep, threads, &mut prof);
            self.packed_dirty = true;
        } else if self.levers.incremental {
            self.sweep_fields(sweep, threads, &mut prof);
        } else {
            self.sweep_scalar(sweep, threads, &mut prof);
        }
        prof.sweeps = 1;
        prof.wall_ns = wall.elapsed().as_nanos();
        prof.threads_used = threads;
        self.sweep_idx += 1;
        prof
    }

    /// `n` sweeps. The returned profile sums them.
    pub fn sweeps(&mut self, n: usize) -> SweepProfile {
        let mut total = SweepProfile::default();
        for _ in 0..n {
            let one = self.sweep();
            total.sweeps += one.sweeps;
            total.wall_ns += one.wall_ns;
            total.add(&one);
            total.threads_used = one.threads_used;
        }
        total
    }

    /// Burn `warmup` sweeps, then keep `draws` energies spaced by `thin` sweeps.
    ///
    /// The state at the end of burn-in is not kept. That is the same window as
    /// [`crate::gibbs::Sampler::collect`].
    pub fn energies(&mut self, warmup: usize, draws: usize, thin: usize) -> Vec<f64> {
        self.sweeps(warmup);
        let thin = thin.max(1);
        let mut out = Vec::with_capacity(draws);
        for _ in 0..draws {
            self.sweeps(thin);
            out.push(self.energy());
        }
        out
    }

    fn sync_spins(&mut self) {
        if self.packed_dirty {
            unpack_all(&self.bits, &mut self.spins, self.side);
            self.packed_dirty = false;
        }
    }

    fn sweep_scalar(&mut self, sweep: u64, threads: usize, prof: &mut SweepProfile) {
        for color in 0..2 {
            let side = self.side;
            let j = self.j;
            let beta = self.beta;
            let thresh = self.thresh;
            let seed = self.seed;
            let tiled = self.levers.tiled;
            let sparse = self.levers.sparse_rng;
            let profile = self.profile;
            let ptr = self.spins.as_mut_ptr() as usize;
            let band = if threads == 1 {
                side
            } else {
                side.div_ceil(threads)
            };
            let acc = std::thread::scope(|scope| {
                let mut joins = Vec::with_capacity(threads);
                for t in 0..threads {
                    let y0 = t * band;
                    if y0 >= side {
                        break;
                    }
                    let y1 = (y0 + band).min(side);
                    joins.push(scope.spawn(move || {
                        let mut local = SweepProfile::default();
                        // SAFETY: each thread owns a disjoint row range of this colour. Sites of
                        // one colour share no edge, so a write never reads a spin another thread
                        // of this colour is writing. The other colour is only read.
                        unsafe {
                            scalar_rows(
                                ptr as *mut i8,
                                side,
                                j,
                                beta,
                                &thresh,
                                seed,
                                sweep,
                                color,
                                y0,
                                y1,
                                tiled,
                                sparse,
                                profile,
                                &mut local,
                            );
                        }
                        local
                    }));
                }
                let mut acc = SweepProfile::default();
                for jn in joins {
                    acc.add(&jn.join().expect("grid worker"));
                }
                acc
            });
            prof.add(&acc);
        }
    }

    fn sweep_fields(&mut self, sweep: u64, threads: usize, prof: &mut SweepProfile) {
        for color in 0..2 {
            let side = self.side;
            let thresh = self.thresh;
            let seed = self.seed;
            let tiled = self.levers.tiled;
            let profile = self.profile;
            let spins = self.spins.as_mut_ptr() as usize;
            let fields = self.fields.as_ptr() as usize;
            let band = if threads == 1 {
                side
            } else {
                side.div_ceil(threads)
            };
            let acc = std::thread::scope(|scope| {
                let mut joins = Vec::new();
                for t in 0..threads {
                    let y0 = t * band;
                    if y0 >= side {
                        break;
                    }
                    let y1 = (y0 + band).min(side);
                    joins.push(scope.spawn(move || {
                        let mut local = SweepProfile::default();
                        // SAFETY: rows y0..y1 of this colour are disjoint. The field vector is
                        // only read. Neighbour fields belong to the other colour and are refreshed
                        // after every thread has finished this colour.
                        unsafe {
                            field_rows(
                                spins as *mut i8,
                                fields as *const i8,
                                side,
                                &thresh,
                                seed,
                                sweep,
                                color,
                                y0,
                                y1,
                                tiled,
                                profile,
                                &mut local,
                            );
                        }
                        local
                    }));
                }
                let mut acc = SweepProfile::default();
                for jn in joins {
                    acc.add(&jn.join().expect("grid field worker"));
                }
                acc
            });
            prof.add(&acc);
            let mut tfield = clock(self.profile);
            refresh_color_fields(&self.spins, &mut self.fields, self.side, color ^ 1);
            stop_clock(&mut tfield, &mut prof.field_ns);
        }
    }

    fn sweep_simd(&mut self, sweep: u64, threads: usize, prof: &mut SweepProfile) {
        for color in 0..2 {
            let side = self.side;
            let seed = self.seed;
            let thresh = self.thresh;
            let sparse = self.levers.sparse_rng;
            let tiled = self.levers.tiled;
            let profile = self.profile;
            let fast_field = self.levers.incremental;
            let write_ptr = self.bits[color].as_mut_ptr() as usize;
            let read_ptr = self.bits[color ^ 1].as_ptr() as usize;
            let band = if threads == 1 {
                side
            } else {
                side.div_ceil(threads)
            };
            let acc = std::thread::scope(|scope| {
                let mut joins = Vec::new();
                for t in 0..threads {
                    let y0 = t * band;
                    if y0 >= side {
                        break;
                    }
                    let y1 = (y0 + band).min(side);
                    joins.push(scope.spawn(move || {
                        let mut local = SweepProfile::default();
                        // SAFETY: `y0..y1` are disjoint. Each thread writes only `write_ptr.add(y)`
                        // for y in that range, and reads the other colour, which this phase does
                        // not write. The outer `bits` vector is not resized.
                        unsafe {
                            simd_rows(
                                write_ptr as *mut Vec<u64>,
                                read_ptr as *const Vec<u64>,
                                side,
                                &thresh,
                                seed,
                                sweep,
                                color,
                                y0,
                                y1,
                                tiled,
                                sparse,
                                fast_field,
                                profile,
                                &mut local,
                            );
                        }
                        local
                    }));
                }
                let mut acc = SweepProfile::default();
                for jn in joins {
                    acc.add(&jn.join().expect("grid simd worker"));
                }
                acc
            });
            prof.add(&acc);
        }
    }
}

/// `E = −J` times the open-grid bond sum.
#[must_use]
pub fn grid_energy(spins: &[i8], side: usize, j: f64) -> f64 {
    let mut e = 0.0;
    for y in 0..side {
        for x in 0..side {
            let si = f64::from(spins[y * side + x]);
            if x + 1 < side {
                e -= j * si * f64::from(spins[y * side + x + 1]);
            }
            if y + 1 < side {
                e -= j * si * f64::from(spins[(y + 1) * side + x]);
            }
        }
    }
    e
}

/// Neighbour-spin sum at `(x, y)`. Missing neighbours contribute 0, not `−1`.
#[must_use]
pub fn neighbor_sum(spins: &[i8], side: usize, x: usize, y: usize) -> i32 {
    let mut f = 0i32;
    if x > 0 {
        f += i32::from(spins[y * side + x - 1]);
    }
    if x + 1 < side {
        f += i32::from(spins[y * side + x + 1]);
    }
    if y > 0 {
        f += i32::from(spins[(y - 1) * side + x]);
    }
    if y + 1 < side {
        f += i32::from(spins[(y + 1) * side + x]);
    }
    f
}

/// `P(s = +1)` for a neighbour-spin sum. This is the only place the grid multiplies `J` into the field.
#[inline]
#[must_use]
pub fn accept_probability(j: f64, beta: f64, field_units: i32) -> f64 {
    p_up(j * f64::from(field_units), beta)
}

/// Colour and the index of the site within its row. `colour` is `(x + y) % 2`.
#[must_use]
pub fn checkerboard(x: usize, y: usize) -> (usize, usize) {
    let color = (x + y) & 1;
    let k = x / 2;
    (color, k)
}

/// Geometric `x` of local index `k` on row `y` of `color`.
#[must_use]
pub fn x_of(y: usize, color: usize, k: usize) -> usize {
    let x_parity = color ^ (y & 1);
    2 * k + x_parity
}

/// How many sites of `color` sit on row `y`.
#[must_use]
pub fn row_sites(side: usize, y: usize, color: usize) -> usize {
    let x_parity = color ^ (y & 1);
    if x_parity >= side {
        0
    } else {
        (side - x_parity).div_ceil(2)
    }
}

/// `round(p_up(J * field, beta) * 2^32)`, clamped to `u32::MAX`. Index is `field + 4`.
#[must_use]
pub fn threshold_table(j: f64, beta: f64) -> [u32; 9] {
    let mut t = [0u32; 9];
    for c in -4..=4 {
        let p = accept_probability(j, beta, c);
        let scaled = (p * 4_294_967_296.0).round();
        t[(c + 4) as usize] = if scaled >= 4_294_967_296.0 {
            u32::MAX
        } else if scaled <= 0.0 {
            0
        } else {
            scaled as u32
        };
    }
    t
}

fn row_rng(seed: u64, sweep: u64, color: usize, y: usize) -> Pcg {
    Pcg::new(
        seed ^ sweep.wrapping_mul(0x9E37_79B9_7F4A_7C15),
        ((y as u64) << 1) | color as u64,
    )
}

/// # Safety
///
/// `spins` points at `side * side` initialised `i8`s. Rows `y0..y1` of `color` are not written by
/// any other caller for the duration of this call. The other colour is only read.
unsafe fn scalar_rows(
    spins: *mut i8,
    side: usize,
    j: f64,
    beta: f64,
    thresh: &[u32; 9],
    seed: u64,
    sweep: u64,
    color: usize,
    y0: usize,
    y1: usize,
    tiled: bool,
    sparse: bool,
    profile: bool,
    prof: &mut SweepProfile,
) {
    let step = if tiled { TILE_ROWS } else { (y1 - y0).max(1) };
    let widest = (y0..y1)
        .map(|y| row_sites(side, y, color))
        .max()
        .unwrap_or(0);
    let mut fields = vec![0i32; widest];
    let mut probs = vec![0.0f64; widest];
    let mut y_base = y0;
    while y_base < y1 {
        let y_hi = (y_base + step).min(y1);
        for y in y_base..y_hi {
            let n = row_sites(side, y, color);
            let mut tmem = clock(profile);
            for k in 0..n {
                let x = x_of(y, color, k);
                fields[k] = unsafe { neighbor_sum_ptr(spins, side, x, y) };
            }
            stop_clock(&mut tmem, &mut prof.memory_ns);
            let mut tfield = clock(profile);
            if !sparse {
                for k in 0..n {
                    probs[k] = accept_probability(j, beta, fields[k]);
                }
            }
            stop_clock(&mut tfield, &mut prof.field_ns);
            let mut rng = row_rng(seed, sweep, color, y);
            let mut trng = clock(profile);
            if sparse {
                let mut k = 0;
                while k < n {
                    let g = (n - k).min(RNG_GROUP);
                    let mut ts = [0u32; RNG_GROUP];
                    for s in 0..g {
                        let c = fields[k + s];
                        ts[s] = thresh[(c + 4) as usize];
                    }
                    let up = decide_group(&ts, g, &mut rng, &mut prof.rng_words);
                    for s in 0..g {
                        let x = x_of(y, color, k + s);
                        unsafe {
                            *spins.add(y * side + x) = if up[s] { 1 } else { -1 };
                        }
                    }
                    k += g;
                }
            } else {
                for k in 0..n {
                    let x = x_of(y, color, k);
                    // One f64 draw consumes two u32s, which is one 64-bit word.
                    let draw = rng.f64();
                    prof.rng_words += 1;
                    unsafe {
                        *spins.add(y * side + x) = if draw < probs[k] { 1 } else { -1 };
                    }
                }
            }
            stop_clock(&mut trng, &mut prof.rng_ns);
        }
        y_base = y_hi;
    }
}

fn refresh_color_fields(spins: &[i8], fields: &mut [i8], side: usize, color: usize) {
    for y in 0..side {
        let n = row_sites(side, y, color);
        for k in 0..n {
            let x = x_of(y, color, k);
            fields[y * side + x] = neighbor_sum(spins, side, x, y) as i8;
        }
    }
}

/// # Safety
///
/// `spins` is `side * side` initialised spins. Rows `y0..y1` of `color` are written only by this
/// call. `fields` is only read.
unsafe fn field_rows(
    spins: *mut i8,
    fields: *const i8,
    side: usize,
    thresh: &[u32; 9],
    seed: u64,
    sweep: u64,
    color: usize,
    y0: usize,
    y1: usize,
    tiled: bool,
    profile: bool,
    prof: &mut SweepProfile,
) {
    let step = if tiled { TILE_ROWS } else { (y1 - y0).max(1) };
    let mut y_base = y0;
    while y_base < y1 {
        let y_hi = (y_base + step).min(y1);
        for y in y_base..y_hi {
            let n = row_sites(side, y, color);
            let mut rng = row_rng(seed, sweep, color, y);
            let mut tmem = clock(profile);
            let mut kept = vec![0i8; n];
            for k in 0..n {
                let x = x_of(y, color, k);
                kept[k] = unsafe { *fields.add(y * side + x) };
            }
            stop_clock(&mut tmem, &mut prof.memory_ns);
            let mut trng = clock(profile);
            for k in 0..n {
                let x = x_of(y, color, k);
                let f = kept[k] as i32;
                let draw = rng.next_u32();
                prof.rng_words += 1;
                let up = draw < thresh[(f + 4) as usize];
                unsafe {
                    *spins.add(y * side + x) = if up { 1 } else { -1 };
                }
            }
            stop_clock(&mut trng, &mut prof.rng_ns);
        }
        y_base = y_hi;
    }
}

/// # Safety
///
/// `spins` points at `side * side` initialised spins. Only reads.
unsafe fn neighbor_sum_ptr(spins: *const i8, side: usize, x: usize, y: usize) -> i32 {
    let mut f = 0i32;
    if x > 0 {
        f += i32::from(unsafe { *spins.add(y * side + x - 1) });
    }
    if x + 1 < side {
        f += i32::from(unsafe { *spins.add(y * side + x + 1) });
    }
    if y > 0 {
        f += i32::from(unsafe { *spins.add((y - 1) * side + x) });
    }
    if y + 1 < side {
        f += i32::from(unsafe { *spins.add((y + 1) * side + x) });
    }
    f
}

fn clock(on: bool) -> Option<std::time::Instant> {
    if on {
        Some(std::time::Instant::now())
    } else {
        None
    }
}

fn stop_clock(t: &mut Option<std::time::Instant>, acc: &mut u128) {
    if let Some(start) = t.take() {
        *acc += start.elapsed().as_nanos();
    }
}

/// Eight sites, high bit first. One `u64` carries the next eight bit-planes (eight bits each).
///
/// `up[s]` is true when the uniform integer is strictly below `thresh[s]`. Unresolved ties
/// (all 32 bits equal) stay down, which is `draw < thresh` and not `draw <= thresh`.
pub fn decide_group(
    thresh: &[u32; RNG_GROUP],
    n: usize,
    rng: &mut Pcg,
    words: &mut u64,
) -> [bool; RNG_GROUP] {
    let mut up = [false; RNG_GROUP];
    let mut open = if n >= 8 { 0xffu8 } else { (1u8 << n) - 1 };
    let mut b = 31i32;
    while b >= 0 && open != 0 {
        let word = rng.next_u64();
        *words += 1;
        for plane in 0..8 {
            if b < 0 || open == 0 {
                break;
            }
            for s in 0..n {
                let bit = 1u8 << s;
                if open & bit == 0 {
                    continue;
                }
                // High bits of the PCG word. The low bits of nearby streams are serially
                // correlated, and using them as the leading comparison bit biases the chain
                // even when a planted-state marginal still looks right.
                let r = ((word >> (63 - (s * 8 + plane))) & 1) as u32;
                let t = (thresh[s] >> (b as u32)) & 1;
                if r != t {
                    up[s] = r == 0 && t == 1;
                    open &= !bit;
                }
            }
            b -= 1;
        }
    }
    up
}

/// # Safety
///
/// `write` and `read` each point at `side` rows. This call writes only rows `y0..y1` of `write`
/// and does not resize those vectors' callers. `read` is not written.
/// Popcount of four bit-planes. Lane value is in `0..=4`, little-endian in the three words.
fn sum4(a: u64, b: u64, c: u64, d: u64) -> [u64; 3] {
    let s0 = a ^ b;
    let s1 = a & b;
    let t0 = s0 ^ c;
    let t1 = s1 ^ (s0 & c);
    let t2 = s1 & s0 & c;
    let u0 = t0 ^ d;
    let u1 = t1 ^ (t0 & d);
    let u2 = t2 ^ (t1 & t0 & d);
    [u0, u1, u2]
}

fn sub_from_plane(acc: &mut [u64; 4], plane: usize, mut bits: u64) {
    for p in plane..4 {
        if bits == 0 {
            return;
        }
        let next = !acc[p] & bits;
        acc[p] ^= bits;
        bits = next;
    }
}

/// Neighbour-spin sum, `-4..=4`, as 4-bit two's complement planes.
///
/// Each neighbour is a bit (`1` is `+1`) and a present mask. A missing neighbour adds nothing.
/// The count of `+1` neighbours minus the count of `−1` neighbours is a popcount of those planes,
/// not a fresh walk of four scalar loads and not an exponential.
fn field_planes(
    up: u64,
    up_p: u64,
    down: u64,
    down_p: u64,
    left: u64,
    left_p: u64,
    right: u64,
    right_p: u64,
) -> [u64; 4] {
    let plus = sum4(up & up_p, down & down_p, left & left_p, right & right_p);
    let minus = sum4(!up & up_p, !down & down_p, !left & left_p, !right & right_p);
    let mut acc = [plus[0], plus[1], plus[2], 0];
    sub_from_plane(&mut acc, 0, minus[0]);
    sub_from_plane(&mut acc, 1, minus[1]);
    sub_from_plane(&mut acc, 2, minus[2]);
    acc
}

unsafe fn simd_rows(
    write: *mut Vec<u64>,
    read: *const Vec<u64>,
    side: usize,
    thresh: &[u32; 9],
    seed: u64,
    sweep: u64,
    color: usize,
    y0: usize,
    y1: usize,
    tiled: bool,
    sparse: bool,
    fast_field: bool,
    profile: bool,
    prof: &mut SweepProfile,
) {
    let step = if tiled { TILE_ROWS } else { (y1 - y0).max(1) };
    let widest = (y0..y1)
        .map(|y| row_sites(side, y, color).div_ceil(SIMD_LANES))
        .max()
        .unwrap_or(0);
    let mut left = vec![0u64; widest];
    let mut right = vec![0u64; widest];
    let mut left_p = vec![0u64; widest];
    let mut right_p = vec![0u64; widest];
    let mut up_b = vec![0u64; widest];
    let mut up_p = vec![0u64; widest];
    let mut dn_b = vec![0u64; widest];
    let mut dn_p = vec![0u64; widest];
    let mut accs = vec![[0u64; 4]; widest];
    let mut y_base = y0;
    while y_base < y1 {
        let y_hi = (y_base + step).min(y1);
        for y in y_base..y_hi {
            let nbits = row_sites(side, y, color);
            let missing_left = (y + color).is_multiple_of(2);
            let nwords = nbits.div_ceil(SIMD_LANES);
            let mut tmem = clock(profile);
            for w in 0..nwords {
                let lo = w * SIMD_LANES;
                let keep = (nbits - lo).min(SIMD_LANES);
                let valid = if keep == 64 {
                    u64::MAX
                } else {
                    (1u64 << keep) - 1
                };
                let same = unsafe { &*read.add(y) };
                let h = horizontal(same, nbits, lo, valid, missing_left);
                left[w] = h.left;
                right[w] = h.right;
                left_p[w] = h.left_present;
                right_p[w] = h.right_present;
                let up_row = if y > 0 {
                    Some(unsafe { &*read.add(y - 1) }.as_slice())
                } else {
                    None
                };
                let (ub, up) = vertical(up_row, lo, nbits, valid);
                up_b[w] = ub;
                up_p[w] = up;
                let down_row = if y + 1 < side {
                    Some(unsafe { &*read.add(y + 1) }.as_slice())
                } else {
                    None
                };
                let (db, dp) = vertical(down_row, lo, nbits, valid);
                dn_b[w] = db;
                dn_p[w] = dp;
            }
            stop_clock(&mut tmem, &mut prof.memory_ns);
            let mut tfield = clock(profile);
            for w in 0..nwords {
                accs[w] = if fast_field {
                    field_planes(
                        up_b[w], up_p[w], dn_b[w], dn_p[w], left[w], left_p[w], right[w],
                        right_p[w],
                    )
                } else {
                    let mut acc = [0u64; 4];
                    add_trit(&mut acc, up_b[w], up_p[w]);
                    add_trit(&mut acc, dn_b[w], dn_p[w]);
                    add_trit(&mut acc, left[w], left_p[w]);
                    add_trit(&mut acc, right[w], right_p[w]);
                    acc
                };
            }
            stop_clock(&mut tfield, &mut prof.field_ns);
            let mut rng = row_rng(seed, sweep, color, y);
            let mut trng = clock(profile);
            for w in 0..nwords {
                let lo = w * SIMD_LANES;
                let keep = (nbits - lo).min(SIMD_LANES);
                let valid = if keep == 64 {
                    u64::MAX
                } else {
                    (1u64 << keep) - 1
                };
                let up = decide_word(
                    &accs[w],
                    valid,
                    thresh,
                    &mut rng,
                    sparse,
                    &mut prof.rng_words,
                );
                let idx = lo / SIMD_LANES;
                let row = unsafe { &mut *write.add(y) };
                let cur = row[idx];
                row[idx] = (cur & !valid) | (up & valid);
            }
            stop_clock(&mut trng, &mut prof.rng_ns);
        }
        y_base = y_hi;
    }
}

fn decide_word(
    acc: &[u64; 4],
    valid: u64,
    thresh: &[u32; 9],
    rng: &mut Pcg,
    sparse: bool,
    words: &mut u64,
) -> u64 {
    let mut masks = [0u64; 9];
    for c in -4..=4 {
        masks[(c + 4) as usize] = mask_eq(acc, c) & valid;
    }
    let mut up = 0u64;
    let mut open = valid;
    for b in (0..32).rev() {
        if sparse && open == 0 {
            break;
        }
        let rand = rng.next_u64().reverse_bits();
        *words += 1;
        if open == 0 {
            continue;
        }
        let mut tbit = 0u64;
        for (i, m) in masks.iter().enumerate() {
            if (thresh[i] >> b) & 1 == 1 {
                tbit |= *m;
            }
        }
        let diff = (rand ^ tbit) & open;
        up |= diff & !rand;
        open &= !diff;
    }
    up
}

fn mask_eq(acc: &[u64; 4], c: i32) -> u64 {
    let bits = (c as u8) & 0x0f;
    let mut m = !0u64;
    for p in 0..4 {
        let plane = acc[p];
        m &= if (bits >> p) & 1 == 1 { plane } else { !plane };
    }
    m
}

fn add_trit(acc: &mut [u64; 4], bits: u64, present: u64) {
    add_bit(acc, bits & present);
    sub_bit(acc, !bits & present);
}

fn add_bit(acc: &mut [u64; 4], mut x: u64) {
    for plane in acc.iter_mut() {
        if x == 0 {
            return;
        }
        let next = *plane & x;
        *plane ^= x;
        x = next;
    }
}

fn sub_bit(acc: &mut [u64; 4], mut x: u64) {
    for plane in acc.iter_mut() {
        if x == 0 {
            return;
        }
        let next = !*plane & x;
        *plane ^= x;
        x = next;
    }
}

struct HBits {
    left: u64,
    right: u64,
    left_present: u64,
    right_present: u64,
}

fn horizontal(other: &[u64], nbits: usize, lo: usize, valid: u64, missing_left: bool) -> HBits {
    if missing_left {
        let left = extract64(other, lo as isize - 1, nbits);
        let right = extract64(other, lo as isize, nbits);
        let mut left_present = valid;
        if lo == 0 {
            left_present &= !1;
        }
        HBits {
            left,
            right,
            left_present,
            right_present: valid,
        }
    } else {
        let left = extract64(other, lo as isize, nbits);
        let right = extract64(other, lo as isize + 1, nbits);
        let mut right_present = valid;
        if nbits > 0 {
            let last = nbits - 1;
            if last >= lo && last - lo < 64 {
                right_present &= !(1u64 << (last - lo));
            }
        }
        HBits {
            left,
            right,
            left_present: valid,
            right_present,
        }
    }
}

fn vertical(row: Option<&[u64]>, lo: usize, nbits: usize, valid: u64) -> (u64, u64) {
    match row {
        None => (0, 0),
        Some(words) => (extract64(words, lo as isize, nbits), valid),
    }
}

fn extract64(words: &[u64], start: isize, nbits: usize) -> u64 {
    if nbits == 0 || start >= nbits as isize || words.is_empty() {
        return 0;
    }
    if start < 0 {
        let mut out = 0u64;
        for i in 0..64 {
            let src = start + i as isize;
            if src >= 0 && (src as usize) < nbits && bit_at(words, src as usize) {
                out |= 1u64 << i;
            }
        }
        return out;
    }
    let start_u = start as usize;
    let wi = start_u / 64;
    let sh = start_u % 64;
    if wi >= words.len() {
        return 0;
    }
    let mut w = words[wi] >> sh;
    if sh != 0 && wi + 1 < words.len() {
        w |= words[wi + 1] << (64 - sh);
    }
    let keep = (nbits - start_u).min(64);
    if keep < 64 {
        w &= (1u64 << keep) - 1;
    }
    w
}

fn bit_at(words: &[u64], i: usize) -> bool {
    ((words[i / 64] >> (i % 64)) & 1) == 1
}

fn pack_all(spins: &[i8], side: usize) -> Vec<Vec<Vec<u64>>> {
    let mut bits = vec![vec![Vec::new(); side]; 2];
    for color in 0..2 {
        for y in 0..side {
            bits[color][y] = pack_row(spins, side, y, color);
        }
    }
    bits
}

fn pack_row(spins: &[i8], side: usize, y: usize, color: usize) -> Vec<u64> {
    let nbits = row_sites(side, y, color);
    if nbits == 0 {
        return Vec::new();
    }
    let mut words = vec![0u64; nbits.div_ceil(64)];
    for k in 0..nbits {
        let x = x_of(y, color, k);
        if spins[y * side + x] == 1 {
            words[k / 64] |= 1u64 << (k % 64);
        }
    }
    words
}

fn unpack_all(bits: &[Vec<Vec<u64>>], spins: &mut [i8], side: usize) {
    for color in 0..2 {
        for y in 0..side {
            let nbits = row_sites(side, y, color);
            for k in 0..nbits {
                let x = x_of(y, color, k);
                let words = &bits[color][y];
                let bit = (words[k / 64] >> (k % 64)) & 1;
                spins[y * side + x] = if bit == 1 { 1 } else { -1 };
            }
        }
    }
}

/// Bit-sliced neighbour sums for one colour, read off `spins` packed the way the sweep packs them.
///
/// Test aid. The sweep uses the same `horizontal` / `vertical` / trit adder.
#[must_use]
pub fn packed_neighbor_sums(spins: &[i8], side: usize, color: usize) -> Vec<Vec<i32>> {
    let bits = pack_all(spins, side);
    let read = if color == 0 { &bits[1] } else { &bits[0] };
    let mut rows = Vec::with_capacity(side);
    for y in 0..side {
        let nbits = row_sites(side, y, color);
        let missing_left = (y + color).is_multiple_of(2);
        let nwords = nbits.div_ceil(SIMD_LANES).max(1);
        let mut sums = vec![0i32; nbits];
        if nbits == 0 {
            rows.push(sums);
            continue;
        }
        for w in 0..nwords {
            let lo = w * SIMD_LANES;
            if lo >= nbits {
                break;
            }
            let keep = (nbits - lo).min(SIMD_LANES);
            let valid = if keep == 64 {
                u64::MAX
            } else {
                (1u64 << keep) - 1
            };
            let h = horizontal(&read[y], nbits, lo, valid, missing_left);
            let (ub, up) = vertical(
                if y > 0 {
                    Some(read[y - 1].as_slice())
                } else {
                    None
                },
                lo,
                nbits,
                valid,
            );
            let (db, dp) = vertical(
                if y + 1 < side {
                    Some(read[y + 1].as_slice())
                } else {
                    None
                },
                lo,
                nbits,
                valid,
            );
            let mut acc = [0u64; 4];
            add_trit(&mut acc, ub, up);
            add_trit(&mut acc, db, dp);
            add_trit(&mut acc, h.left, h.left_present);
            add_trit(&mut acc, h.right, h.right_present);
            for lane in 0..keep {
                sums[lo + lane] = lane_i4(&acc, lane);
            }
        }
        rows.push(sums);
    }
    rows
}

fn lane_i4(acc: &[u64; 4], lane: usize) -> i32 {
    let mut u = 0u8;
    for p in 0..4 {
        if (acc[p] >> lane) & 1 == 1 {
            u |= 1 << p;
        }
    }
    if u & 0x8 != 0 {
        i32::from(u) - 16
    } else {
        i32::from(u)
    }
}

/// Counter hash shared with the packed checkerboard shader. The whole `u32` is the draw.
/// Low bits of a nearby PCG stream are not the leading bit.
#[must_use]
pub fn packed_hash(seed: u32, site: u32, sweep: u32) -> u32 {
    let mut x = seed
        .wrapping_add(site.wrapping_mul(0x9E37_79B9))
        .wrapping_add(sweep.wrapping_mul(0x85EB_CA6B));
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846C_A68B);
    x ^= x >> 16;
    x
}

/// Both halves of the seed reach the shader. A `u64` truncated to `as u32` would drop the top.
#[must_use]
pub fn packed_seed(seed: u64) -> u32 {
    (seed as u32) ^ ((seed >> 32) as u32)
}

/// Checkerboard Gibbs on spins packed into `u32` words, one bit per site of that colour.
///
/// Even side only. Bit 1 is `+1` and bit 0 is `−1`. Sites of one colour do not share an edge, so
/// a word is updated from the other colour's words and the result does not depend on thread
/// order. The acceptance test is `packed_hash < threshold_table[field + 4]`, the same comparison
/// the shader runs.
///
/// # Errors
///
/// An odd or zero side, a state whose length is not `side²`, or a non-finite coupling or a
/// negative temperature.
pub fn sweep_packed_cpu(
    side: usize,
    spins: &mut [i8],
    j: f64,
    beta: f64,
    seed: u64,
    start_sweep: u32,
    sweeps: u32,
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
    let half = side / 2;
    let nw = half.div_ceil(32);
    let mut plane = [vec![0u32; side * nw], vec![0u32; side * nw]];
    pack_planes(&mut plane, spins, side, nw);
    let thresh = threshold_table(j, beta);
    let seed32 = packed_seed(seed);
    for s in 0..sweeps {
        let sweep = start_sweep.wrapping_add(s);
        for color in 0..2 {
            for y in 0..side {
                for w in 0..nw {
                    let mut word = plane[color][y * nw + w];
                    let k0 = w * 32;
                    for b in 0..32 {
                        let k = k0 + b;
                        if k >= half {
                            break;
                        }
                        let x = 2 * k + ((y + color) & 1);
                        let field = packed_field(&plane, side, nw, color, x, y);
                        let site = (y * side + x) as u32;
                        let draw = packed_hash(seed32, site, sweep);
                        let up = draw < thresh[(field + 4) as usize];
                        let mask = 1u32 << b;
                        if up {
                            word |= mask;
                        } else {
                            word &= !mask;
                        }
                    }
                    plane[color][y * nw + w] = word;
                }
            }
        }
    }
    unpack_planes(&plane, spins, side, nw);
    Ok(())
}

fn pack_planes(plane: &mut [Vec<u32>; 2], spins: &[i8], side: usize, nw: usize) {
    for y in 0..side {
        for x in 0..side {
            if spins[y * side + x] > 0 {
                let (color, k) = checkerboard(x, y);
                plane[color][y * nw + k / 32] |= 1u32 << (k % 32);
            }
        }
    }
}

fn unpack_planes(plane: &[Vec<u32>; 2], spins: &mut [i8], side: usize, nw: usize) {
    for y in 0..side {
        for x in 0..side {
            let (color, k) = checkerboard(x, y);
            let word = plane[color][y * nw + k / 32];
            spins[y * side + x] = if (word >> (k % 32)) & 1 == 1 { 1 } else { -1 };
        }
    }
}

fn packed_spin(words: &[u32], side: usize, nw: usize, color: usize, x: usize, y: usize) -> i32 {
    if y >= side || x >= side {
        return 0;
    }
    let parity = (y + color) & 1;
    if (x & 1) != parity {
        return 0;
    }
    let k = (x - parity) / 2;
    let word = words[y * nw + k / 32];
    if (word >> (k % 32)) & 1 == 1 { 1 } else { -1 }
}

fn packed_field(
    plane: &[Vec<u32>; 2],
    side: usize,
    nw: usize,
    color: usize,
    x: usize,
    y: usize,
) -> i32 {
    let other = color ^ 1;
    let words = plane[other].as_slice();
    let mut sum = 0;
    if y > 0 {
        sum += packed_spin(words, side, nw, other, x, y - 1);
    }
    if y + 1 < side {
        sum += packed_spin(words, side, nw, other, x, y + 1);
    }
    if x > 0 {
        sum += packed_spin(words, side, nw, other, x - 1, y);
    }
    if x + 1 < side {
        sum += packed_spin(words, side, nw, other, x + 1, y);
    }
    sum
}

/// WGSL for [`sweep_packed_cpu`]. The gpu crate compiles this string. A second copy of the
/// update rule would drift the first time one side was tuned.
#[must_use]
pub fn packed_checkerboard_shader() -> &'static str {
    PACKED_SHADER
}

const PACKED_SHADER: &str = r#"
struct Params {
  side: u32,
  nwords: u32,
  color: u32,
  sweep: u32,
  seed: u32,
  half: u32,
  pad0: u32,
  pad1: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> mine: array<u32>;
@group(0) @binding(2) var<storage, read> other: array<u32>;
@group(0) @binding(3) var<storage, read> thresh: array<u32>;

fn packed_hash(seed: u32, site: u32, sweep: u32) -> u32 {
  var x = seed + site * 0x9E3779B9u + sweep * 0x85EBCA6Bu;
  x = x ^ (x >> 16u);
  x = x * 0x7FEB352Du;
  x = x ^ (x >> 15u);
  x = x * 0x846CA68Bu;
  x = x ^ (x >> 16u);
  return x;
}

fn spin_at(x: i32, y: i32) -> i32 {
  if (y < 0 || x < 0 || u32(y) >= p.side || u32(x) >= p.side) {
    return 0;
  }
  let parity = (u32(y) + (p.color ^ 1u)) & 1u;
  if ((u32(x) & 1u) != parity) {
    return 0;
  }
  let k = (u32(x) - parity) / 2u;
  let word = other[u32(y) * p.nwords + (k / 32u)];
  let bit = (word >> (k % 32u)) & 1u;
  if (bit == 1u) { return 1; }
  return -1;
}

@compute @workgroup_size(64)
fn sweep(@builtin(global_invocation_id) gid: vec3<u32>) {
  let word_i = gid.x;
  let total = p.side * p.nwords;
  if (word_i >= total) { return; }
  let y = word_i / p.nwords;
  let w = word_i % p.nwords;
  var word = mine[word_i];
  let k0 = w * 32u;
  for (var b: u32 = 0u; b < 32u; b = b + 1u) {
    let k = k0 + b;
    if (k >= p.half) { break; }
    let x = 2u * k + ((y + p.color) & 1u);
    var f: i32 = 0;
    f = f + spin_at(i32(x), i32(y) - 1);
    f = f + spin_at(i32(x), i32(y) + 1);
    f = f + spin_at(i32(x) - 1, i32(y));
    f = f + spin_at(i32(x) + 1, i32(y));
    let site = y * p.side + x;
    let draw = packed_hash(p.seed, site, p.sweep);
    let up = draw < thresh[u32(f + 4)];
    let mask = 1u << b;
    if (up) { word = word | mask; } else { word = word & ~mask; }
  }
  mine[word_i] = word;
}
"#;

// ---- machine-checked ---------------------------------------------------------------------------
//
// The colour index is the address of a site. A sweep that updates the wrong `x` updates a
// neighbour of the site it thinks it is updating, and the two colours are no longer independent.
#[cfg(kani)]
mod proofs {
    use super::{checkerboard, row_sites, x_of};

    /// For every site of a side in `1..=16`, colour and local index round-trip, the row count
    /// covers the row, and two sites in one row of one colour are not adjacent.
    #[kani::proof]
    fn checkerboard_round_trips_and_a_row_is_an_independent_set() {
        let side: usize = kani::any();
        kani::assume(side >= 1 && side <= 16);
        let x: usize = kani::any();
        let y: usize = kani::any();
        kani::assume(x < side && y < side);
        let (color, k) = checkerboard(x, y);
        assert!(color < 2);
        assert!(k < row_sites(side, y, color));
        assert_eq!(x_of(y, color, k), x);
        let n = row_sites(side, y, color);
        let mut covered = 0usize;
        for kk in 0..n {
            let xx = x_of(y, color, kk);
            assert!(xx < side);
            assert_eq!(checkerboard(xx, y).0, color);
            assert_eq!(checkerboard(xx, y).1, kk);
            if kk > 0 {
                assert!(xx >= x_of(y, color, kk - 1) + 2);
            }
            covered += 1;
        }
        assert_eq!(covered, n);
        let other = color ^ 1;
        assert_eq!(row_sites(side, y, color) + row_sites(side, y, other), side);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ising::{exact_boltzmann, grid2d, tv};

    fn tv_of(levers: Levers, beta: f64, seed: u64, burn: usize, n: usize) -> (f64, f64, f64) {
        let side = 2usize;
        let j = 1.0;
        let mut g = OpenIsing::try_new(side, j, beta, seed, levers, 2).unwrap();
        g.sweeps(burn);
        let graph = grid2d(side, side, j);
        let exact = exact_boltzmann(&graph, beta);
        let mut hist = vec![0.0f64; exact.len()];
        for _ in 0..n {
            g.sweep();
            let mut mask = 0usize;
            for (b, &s) in g.spins().iter().enumerate() {
                if s == 1 {
                    mask |= 1 << b;
                }
            }
            hist[mask] += 1.0;
        }
        for h in &mut hist {
            *h /= n as f64;
        }
        let mut mean = 0.0;
        let mut exact_mean = 0.0;
        let mut s = vec![-1i8; 4];
        for mask in 0..hist.len() {
            for b in 0..4 {
                s[b] = if mask >> b & 1 == 1 { 1 } else { -1 };
            }
            let e = grid_energy(&s, 2, 1.0);
            mean += hist[mask] * e;
            exact_mean += exact[mask] * e;
        }
        (tv(&hist, &exact), mean, exact_mean)
    }

    #[test]
    fn each_lever_matches_the_boltzmann_law_on_a_2x2() {
        let configs = [
            Levers::SCALAR,
            Levers {
                parallel: true,
                ..Levers::SCALAR
            },
            Levers {
                simd: true,
                ..Levers::SCALAR
            },
            Levers {
                tiled: true,
                ..Levers::SCALAR
            },
            Levers {
                sparse_rng: true,
                ..Levers::SCALAR
            },
            Levers {
                parallel: true,
                simd: true,
                tiled: true,
                sparse_rng: true,
                incremental: false,
            },
            Levers::COMBINED,
            Levers {
                incremental: true,
                ..Levers::SCALAR
            },
        ];
        for (i, levers) in configs.iter().enumerate() {
            let (distance, mean, exact) = tv_of(*levers, 0.3, 0x6D + i as u64, 400, 40_000);
            assert!(
                distance < 0.03,
                "beta 0.3 levers {levers:?} TV {distance} mean {mean} exact {exact}"
            );
        }
        // beta 1.2 puts almost all the mass on the two aligned states, and a chain that
        // starts at all −1 needs a long excursion to visit all +1. The mean is the check
        // that fails if the field sign is wrong.
        for (i, levers) in [
            Levers::SCALAR,
            Levers {
                sparse_rng: true,
                ..Levers::SCALAR
            },
            Levers {
                simd: true,
                ..Levers::SCALAR
            },
        ]
        .iter()
        .enumerate()
        {
            let (distance, mean, exact) = tv_of(*levers, 1.2, 0xB0 + i as u64, 2_000, 20_000);
            assert!(
                (mean - exact).abs() < 0.08,
                "beta 1.2 levers {levers:?} mean {mean} exact {exact} TV {distance}"
            );
        }
    }

    #[test]
    fn the_same_seed_and_levers_are_bit_identical_and_threads_do_not_move_the_sample() {
        let levers = Levers {
            parallel: true,
            tiled: true,
            ..Levers::SCALAR
        };
        let mut a = OpenIsing::try_new(8, 1.0, 0.3, 99, levers, 1).unwrap();
        let mut b = OpenIsing::try_new(8, 1.0, 0.3, 99, levers, 3).unwrap();
        a.sweeps(25);
        b.sweeps(25);
        assert_eq!(a.spins(), b.spins());
        assert_eq!(b.threads_used(), 3);
        let mut c = OpenIsing::try_new(8, 1.0, 0.3, 99, Levers::SCALAR, 4).unwrap();
        c.sweeps(25);
        // Tiling and threading key the stream by row, same as the scalar path.
        assert_eq!(a.spins(), c.spins());
        assert_eq!(c.threads_used(), 1, "parallel off does not spawn");
    }

    #[test]
    fn simd_matches_its_own_repeat_and_odd_sides_are_refused() {
        let mut a = OpenIsing::try_new(
            8,
            1.0,
            0.4,
            7,
            Levers {
                simd: true,
                ..Levers::SCALAR
            },
            2,
        )
        .unwrap();
        let mut b = OpenIsing::try_new(
            8,
            1.0,
            0.4,
            7,
            Levers {
                simd: true,
                sparse_rng: true,
                ..Levers::SCALAR
            },
            2,
        )
        .unwrap();
        a.sweeps(10);
        let first = a.spins().to_vec();
        let mut again = OpenIsing::try_new(
            8,
            1.0,
            0.4,
            7,
            Levers {
                simd: true,
                ..Levers::SCALAR
            },
            2,
        )
        .unwrap();
        again.sweeps(10);
        assert_eq!(first, again.spins());
        // sparse-rng changes the encoding, so it must be allowed to differ, and it must still move.
        b.sweeps(10);
        assert_ne!(b.spins(), &vec![-1i8; 64]);
        let err = match OpenIsing::try_new(
            5,
            1.0,
            0.3,
            1,
            Levers {
                simd: true,
                ..Levers::SCALAR
            },
            1,
        ) {
            Err(err) => err,
            Ok(_) => panic!("simd accepted an odd side"),
        };
        assert!(err.message.contains("even"), "{}", err.message);
        assert!(err.message.contains('5'), "{}", err.message);
    }

    #[test]
    fn popcount_field_matches_a_trit_sum_and_combined_matches_simd() {
        for a in 0..2u64 {
            for b in 0..2u64 {
                for c in 0..2u64 {
                    for d in 0..2u64 {
                        let planes = sum4(a, b, c, d);
                        let got = lane_i4(&[planes[0], planes[1], planes[2], 0], 0);
                        assert_eq!(got, (a + b + c + d) as i32);
                    }
                }
            }
        }
        let mut plain = OpenIsing::try_new(
            8,
            1.0,
            0.4,
            7,
            Levers {
                simd: true,
                ..Levers::SCALAR
            },
            1,
        )
        .unwrap();
        let mut combined = OpenIsing::try_new(8, 1.0, 0.4, 7, Levers::COMBINED, 3).unwrap();
        plain.sweeps(12);
        combined.sweeps(12);
        assert_eq!(
            plain.spins(),
            combined.spins(),
            "popcount field must be the same chain as the trit adder"
        );
        assert_eq!(combined.threads_used(), 3);
        let mut one = OpenIsing::try_new(
            8,
            1.0,
            0.4,
            7,
            Levers {
                incremental: true,
                parallel: true,
                tiled: true,
                ..Levers::SCALAR
            },
            1,
        )
        .unwrap();
        let mut many = OpenIsing::try_new(
            8,
            1.0,
            0.4,
            7,
            Levers {
                incremental: true,
                parallel: true,
                tiled: true,
                ..Levers::SCALAR
            },
            3,
        )
        .unwrap();
        one.sweeps(12);
        many.sweeps(12);
        assert_eq!(
            one.spins(),
            many.spins(),
            "kept fields of one colour are not written by two threads"
        );
    }

    #[test]
    fn packed_words_match_boltzmann_and_repeat_for_a_fixed_seed() {
        let beta = 0.3;
        let exact = exact_boltzmann(&grid2d(2, 2, 1.0), beta);
        let mut hist = vec![0.0; exact.len()];
        let mut state = vec![-1i8; 4];
        let burn = 200u32;
        let draws = 20_000u32;
        sweep_packed_cpu(2, &mut state, 1.0, beta, 0x51, 0, burn).unwrap();
        for i in 0..draws {
            sweep_packed_cpu(2, &mut state, 1.0, beta, 0x51, burn + i, 1).unwrap();
            let mut mask = 0usize;
            for (b, &s) in state.iter().enumerate() {
                if s == 1 {
                    mask |= 1 << b;
                }
            }
            hist[mask] += 1.0;
        }
        for h in &mut hist {
            *h /= f64::from(draws);
        }
        let distance = tv(&hist, &exact);
        assert!(distance < 0.03, "packed checkerboard TV {distance}");
        let mut a = vec![-1i8; 16];
        let mut b = vec![-1i8; 16];
        sweep_packed_cpu(4, &mut a, 1.0, 0.4, 9, 0, 8).unwrap();
        sweep_packed_cpu(4, &mut b, 1.0, 0.4, 9, 0, 8).unwrap();
        assert_eq!(a, b);
        let mut c = vec![-1i8; 16];
        sweep_packed_cpu(4, &mut c, 1.0, 0.4, 10, 0, 8).unwrap();
        assert_ne!(a, c);
        let err = sweep_packed_cpu(3, &mut vec![-1i8; 9], 1.0, 0.3, 1, 0, 1).unwrap_err();
        assert!(err.contains("even"), "{err}");
    }

    #[test]
    fn packed_fields_match_the_scalar_neighbour_sum_including_edges() {
        for side in [2usize, 4, 8, 18] {
            let mut spins = vec![0i8; side * side];
            for y in 0..side {
                for x in 0..side {
                    spins[y * side + x] = if (x * 3 + y * 5).is_multiple_of(2) {
                        1
                    } else {
                        -1
                    };
                }
            }
            for color in 0..2 {
                let packed = packed_neighbor_sums(&spins, side, color);
                for y in 0..side {
                    let n = row_sites(side, y, color);
                    assert_eq!(packed[y].len(), n);
                    for k in 0..n {
                        let x = x_of(y, color, k);
                        let want = neighbor_sum(&spins, side, x, y);
                        assert_eq!(packed[y][k], want, "side {side} color {color} ({x},{y})");
                    }
                }
            }
        }
    }

    #[test]
    fn open_grid_energy_matches_graph_energy() {
        let side = 6usize;
        let j = 1.0;
        let mut spins = vec![1i8; side * side];
        spins[0] = -1;
        spins[side + 2] = -1;
        let graph = grid2d(side, side, j);
        assert!((grid_energy(&spins, side, j) - graph.energy(&spins)).abs() < 1e-9);
        let mut g = OpenIsing::try_new(side, j, 0.3, 1, Levers::SCALAR, 1).unwrap();
        g.sweeps(3);
        let got = g.energy();
        let s = g.spins().to_vec();
        assert!((got - grid2d(side, side, j).energy(&s)).abs() < 1e-9);
    }

    #[test]
    fn grouped_draws_keep_independent_exact_thresholds() {
        // p reconstructed from the table, not from a shared uniform.
        let thresh_of = |p: f64| -> u32 {
            let scaled = (p * 4_294_967_296.0).round();
            if scaled >= 4_294_967_296.0 {
                u32::MAX
            } else {
                scaled as u32
            }
        };
        let p0 = 0.2f64;
        let p1 = 0.8f64;
        let mut ts = [0u32; RNG_GROUP];
        ts[0] = thresh_of(p0);
        ts[1] = thresh_of(p1);
        let mut rng = Pcg::new(12345, 9);
        let n = 20_000usize;
        let mut both = 0usize;
        let mut c0 = 0usize;
        let mut c1 = 0usize;
        let mut words = 0u64;
        for _ in 0..n {
            let up = decide_group(&ts, 2, &mut rng, &mut words);
            if up[0] {
                c0 += 1;
            }
            if up[1] {
                c1 += 1;
            }
            if up[0] && up[1] {
                both += 1;
            }
        }
        let f0 = c0 as f64 / n as f64;
        let f1 = c1 as f64 / n as f64;
        let fb = both as f64 / n as f64;
        assert!((f0 - p0).abs() < 0.02, "site0 {f0}");
        assert!((f1 - p1).abs() < 0.02, "site1 {f1}");
        // A shared uniform would give min(p0, p1) = 0.2, not the product.
        assert!(
            (fb - p0 * p1).abs() < 0.02,
            "joint {fb}, shared-uniform would be near 0.2"
        );
        assert!(
            words < (n as u64) * 8,
            "early exit drew {words} words for {n} pairs"
        );
    }

    #[test]
    fn sparse_one_step_conditional_matches_p_up() {
        let side = 2usize;
        let j = 1.0;
        let beta = 0.3;
        let thresh = threshold_table(j, beta);
        let patterns: [[i8; 4]; 4] = [
            [-1, -1, -1, -1],
            [1, 1, 1, 1],
            [1, -1, -1, 1],
            [-1, 1, 1, -1],
        ];
        let n = 8_000usize;
        for (pi, pattern) in patterns.iter().enumerate() {
            for color in 0..2 {
                let mut hits = vec![0usize; side * side];
                let mut field = vec![0i32; side * side];
                for y in 0..side {
                    for k in 0..row_sites(side, y, color) {
                        let x = x_of(y, color, k);
                        field[y * side + x] = neighbor_sum(pattern, side, x, y);
                    }
                }
                for trial in 0..n {
                    let mut spins = pattern.to_vec();
                    let mut prof = SweepProfile::default();
                    unsafe {
                        scalar_rows(
                            spins.as_mut_ptr(),
                            side,
                            j,
                            beta,
                            &thresh,
                            10 + pi as u64,
                            trial as u64,
                            color,
                            0,
                            side,
                            false,
                            true,
                            false,
                            &mut prof,
                        );
                    }
                    for y in 0..side {
                        for k in 0..row_sites(side, y, color) {
                            let x = x_of(y, color, k);
                            if spins[y * side + x] == 1 {
                                hits[y * side + x] += 1;
                            }
                        }
                    }
                }
                for y in 0..side {
                    for k in 0..row_sites(side, y, color) {
                        let x = x_of(y, color, k);
                        let idx = y * side + x;
                        let p = accept_probability(j, beta, field[idx]);
                        let f = hits[idx] as f64 / n as f64;
                        assert!(
                            (f - p).abs() < 0.03,
                            "pattern {pi} color {color} ({x},{y}) field {} got {f} want {p}",
                            field[idx]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn an_unknown_lever_name_is_an_error_and_zero_threads_is_an_error() {
        let err = Levers::one("cuda").unwrap_err();
        assert!(err.contains("cuda"), "{err}");
        assert!(err.contains("parallel"), "{err}");
        let err = match OpenIsing::try_new(4, 1.0, 0.3, 1, Levers::SCALAR, 0) {
            Err(err) => err,
            Ok(_) => panic!("zero threads was accepted"),
        };
        assert!(err.message.contains('0'), "{}", err.message);
    }

    #[test]
    fn profile_attributes_time_to_the_three_phases_and_off_stays_zero() {
        let mut quiet = OpenIsing::try_new(16, 1.0, 0.3, 1, Levers::SCALAR, 1).unwrap();
        let off = quiet.sweep();
        assert_eq!(off.rng_ns, 0);
        assert_eq!(off.field_ns, 0);
        assert_eq!(off.memory_ns, 0);
        assert!(off.wall_ns > 0);
        assert!(off.rng_words > 0);
        quiet.set_profile(true);
        let on = quiet.sweep();
        assert!(on.memory_ns > 0, "memory {}", on.memory_ns);
        assert!(on.field_ns > 0, "field {}", on.field_ns);
        assert!(on.rng_ns > 0, "rng {}", on.rng_ns);
    }
}
