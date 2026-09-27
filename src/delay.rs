//! Interactions that arrive late: the exact law of a clocked p-bit network whose reads of its
//! neighbours are `d` ticks old.
//!
//! # The claim, and the question it leaves for a fabric
//!
//! Zhang, Gibeault et al. (arXiv:2607.15215, 2026), from coupled superparamagnetic tunnel junctions:
//! *"sufficiently long delays drive the steady-state probabilities toward equal state occupations
//! even in strongly coupled systems"*, and *"delay-induced uniform distributions emerge in a broad
//! class of stochastic networks"*. Their spins flip at an Arrhenius rate that depends on the spin's
//! OWN current state and on its neighbours' states one delay ago, `λ_i ∝ exp(−β f_i(t − τ) s_i(t))`.
//!
//! A p-bit fabric does not all work that way. A heat-bath p-bit draws its new value from its field
//! and ignores the value it holds. This module computes both rules exactly, as a Markov chain on the
//! last `d` frames of the whole network, so the difference is a measurement rather than an argument:
//!
//! * [`Rule::HeatBath`] — every spin redrawn each tick from `P(+1) = σ(2β f_i)`, the field read `d`
//!   ticks back. **The chain splits into `d` interleaved copies of the synchronous sweep that never
//!   meet**, so the equal-time law is Peretto's synchronous law at EVERY `d` — held to
//!   [`crate::autocorr::peretto`] to rounding. Delay neither helps nor washes anything out.
//! * [`Rule::Arrhenius`] — spin `i` flips with probability `p₀ exp(−β f_i s_i)`, `f_i` read `d`
//!   ticks back and `s_i` its own current value: the paper's rule, clocked. Here the copies DO meet,
//!   through each spin's own state, and delay moves the law — toward uniform at zero field, as the
//!   paper says.
//! * [`Rule::Sca`] — the stochastic cellular automaton of [`crate::autocorr::Kernel::Sca`] (STATICA,
//!   Amorphica): every spin redrawn each tick from `P(+1) = σ(2(β f_i / 2 + q s_i))`, the half field
//!   read `d` ticks back and the pinning `q` pulling toward the spin's own CURRENT value. At `d = 1`
//!   it is that kernel, held to [`crate::autocorr::sca_law`] to rounding. The pinning is own-state
//!   dependence, so the delay reaches this rule too -- and the question it answers is an engineering
//!   one: a fabric whose reads arrive late can pin harder or update less often.
//!
//! The chain holds the last `d` frames (the update reads the oldest), `n d` bits, so it is exact
//! only for small networks: 12 bits, `2^12` augmented states, is the cap used here.
//!
//! # The rule that holds for a uniform delay, and the half of it that is proved
//!
//! Let every wire carry the same delay `d` and let every spin redraw on every tick by one rule whose
//! law for the new frame depends on the past only through the frame read:
//! `P(x_{t+1} | x_t, ..., x_{t-d+1}) = K(x_{t+1} | x_{t-d+1})`. Then `x_{t+1}` is a function of
//! `x_{t+1-d}` and that tick's fresh randomness alone, so by induction the ticks of each residue class
//! mod `d`, `x_r, x_{r+d}, x_{r+2d}, ...`, form a Markov chain with kernel `K`, driven by random draws
//! no other class touches: `d` interleaved copies of the one-tick chain that never meet. The current
//! frame has `K`'s stationary law at every `d`. **No dependence on the value a spin holds, no effect
//! of a uniform delay** -- that direction is this paragraph's proof, and [`Rule::HeatBath`] is its
//! instance. The converse, that dependence on the held value lets the delay in, is NOT proved, and it
//! is not true in general (spins with no neighbours hold and redraw with nothing to read late). It is
//! measured: [`Rule::Arrhenius`], [`Rule::Sca`] and the coloured schedule below, each on a few small
//! fixtures at `β = 1` with a clean integer delay on every read.
//!
//! # A coloured fabric read late: [`stationary_coloured`]
//!
//! Colouring is the other repair for a shared clock: on tick `t` class `t mod K` redraws from the
//! heat-bath conditional and every other spin holds. Read fresh (`d = 1`, the frame read is the frame
//! held) it is chromatic Gibbs, exact. But a held spin's next value IS the value it holds, so the
//! schedule has exactly the own-state dependence the rule above says nothing protects, and a late read
//! reaches it. For two coupled spins with classes `{0}, {1}`, at `d >= 2` each draw reads the other
//! spin's value from before that spin's last update -- the draw `L` ticks earlier, `L = 3` for
//! `d = 2, 3` and `5` for `d = 4, 5`, always odd and at least 3. Each draw's dependence runs back
//! along `t, t - L, t - 2L, ...`, alternating between the spins just as the fresh sweep alternates,
//! and the two values in any frame were drawn on consecutive ticks, which lie on different chains
//! unless `L = 1`. So the spins are independent, each with its Boltzmann marginal: at zero field and
//! `β J = 1` the pair agrees 0.8808 of the time read fresh and exactly `1/2` read two or more ticks
//! late. For a pair that product of the marginals is also the synchronous (Peretto) law, by the same
//! interleaving. On PAI-310's frustrated triangle a two-tick read puts the law 0.1616 from Boltzmann
//! (the synchronous law is 0.8139 from it), and the laws after each class differ: 0.1952, 0.1750,
//! 0.1891.
//!
//! # A pinned automaton read late: the law, and what it costs
//!
//! Every flip of the pinned automaton costs `e^{-2q}`, so at large `q` its law is Boltzmann plus a
//! first-order term. Writing `φ_i(x) = e^{-β f_i(x) x_i}`, `Φ = Σ_i φ_i`, and `L` for the generator
//! `(L g)(x) = Σ_i φ_i(x) [g(x) − g(x^i)]` (Boltzmann is its reversible law), the law of the current
//! frame at delay `d` is
//!
//! ```text
//!   π_d = π_G (1 + e^{-2q} g_d) + O(e^{-4q}),   g_d = g_A + (d − ½) g_S,   <g_A> = <g_S> = 0,
//!   L g_A = Σ_i (φ_i² − 1),     L g_S = 4 Σ_{i<j} φ_i φ_j (1 − e^{2β J_ij x_i x_j}),
//! ```
//!
//! so `TV(π_d, π_G) e^{2q} → c_d = E_G|g_d| / 2`, which [`sca_rate_constant`] computes from `2^n`
//! states rather than `2^{n d}`. The `g_S` term is the delay: for `d − 1` ticks after spin `i` flips,
//! each neighbour `j` still flips at its old rate `φ_j(x)` rather than `φ_j(x^i)`, and summing that
//! window's excess flux gives the source `(d − 1) S`. At `d = 1` the algebraic identity
//! `L Φ = Σ_i (φ_i² − 1) + S / 2` makes `g_1 = Φ − <Φ>`, the first-order term of `sca_law`'s proved
//! closed form, so the one-tick automaton already carries half a tick's worth of the delay term.
//! PROVED: the `d = 1` term and that identity. DERIVED, by a first-order expansion that is not a
//! rigorous proof: the `(d − 1) g_S` window term. MEASURED: the exact chain's `e^{2q} TV` at `q = 6`
//! is within `5.1e-5` of `c_d`, relative, on a biased pair, a frustrated triangle and a 4-spin SK
//! instance at every delay the tests build, and within `1e-6` once extrapolated in `e^{-2q}` at every
//! delay up to `n d = 12` (WORKLOADS entry 13).
//!
//! For two spins at zero field, `φ_0 = φ_1`, so `S = 2 Σ_i (φ_i² − 1)` pointwise, `g_S = 2 g_A`, and
//! `c_d = d c_1` exactly at first order: a `d`-tick read costs a pinned pair exactly `d` times the
//! distance. Elsewhere `c_d` is only close to `d c_1` -- piecewise linear in `d`, kinked wherever
//! some `g_d(x)` changes sign -- and the gap decides the engineering choice. To hold the law within
//! `ε` a fabric can PIN harder (`q*(ε, d) ≈ ln(c_d / ε) / 2`) or WAIT, running the one-tick automaton
//! on every `d`-th tick with fresh reads. Moves per tick go as `e^{-2q}`, so pinning moves
//! `d c_1 / c_d` times as many spins per tick as waiting: pinning wins iff `c_d < d c_1`. Either way a
//! `d`-tick read costs about a factor of `d` in rate; see [`stationary_solved`] for the exact count.

use crate::graph::Graph;

/// How each spin draws its next value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rule {
    /// Redraw from the heat-bath conditional at the delayed field, ignoring the current value.
    HeatBath,
    /// Flip with probability `p0 · exp(−β f_i s_i)` (capped at 1), `f_i` delayed, `s_i` current.
    Arrhenius {
        /// Attempt probability per tick at zero field: the clocked `λ Δt`.
        p0: f64,
    },
    /// The stochastic cellular automaton of [`crate::autocorr::Kernel::Sca`]: redraw from
    /// `P(+1) = 1 / (1 + exp(−2 (β f_i / 2 + q s_i)))`, `f_i` delayed and `s_i` current. At `d = 1`
    /// it is exactly that kernel.
    Sca {
        /// The pinning, in units of the exponent: dimensionless, NOT multiplied by `beta`. `q >= 0`.
        q: f64,
    },
}

/// The most frame bits the exact chain is built over, `n d`.
pub const MAX_BITS: usize = 12;

fn spin(x: usize, i: usize) -> i8 {
    if (x >> i) & 1 == 1 {
        1
    } else {
        -1
    }
}

fn frame_spins(x: usize, n: usize) -> Vec<i8> {
    (0..n).map(|i| spin(x, i)).collect()
}

/// Probability that spin `i` is `+1` after one tick, given its current value and the delayed frame.
fn p_plus(g: &Graph, beta: f64, rule: Rule, i: usize, current: i8, delayed: &[i8]) -> f64 {
    let f = g.field(i, delayed);
    match rule {
        Rule::HeatBath => 1.0 / (1.0 + (-2.0 * beta * f).exp()),
        Rule::Arrhenius { p0 } => {
            let flip = (p0 * (-beta * f * f64::from(current)).exp()).min(1.0);
            if current > 0 {
                1.0 - flip
            } else {
                flip
            }
        }
        // `autocorr::p_site`'s arm for `Kernel::Sca`, argument for argument, so that `d = 1` is that
        // kernel to the last bit: the half field (delayed here) plus the pinning times the spin's
        // CURRENT value. Pinning the delayed value instead would make the new value a function of
        // the delayed frame alone, and the chain would split like the heat bath's.
        Rule::Sca { q } => {
            assert!(q.is_finite() && q >= 0.0, "SCA pinning q must be finite and non-negative, got {q}");
            crate::kernel::p_up(0.5 * beta * f + q * f64::from(current), 1.0)
        }
    }
}

/// The stationary law of the network's CURRENT frame, over its `2^n` states (bit `i` set is spin `i`
/// at `+1`), when every read of a neighbour is `d ≥ 1` ticks old. `d = 1` is the ordinary clocked
/// update from the previous frame.
///
/// Built as a chain on the last `d` frames and pushed to stationarity by power iteration from the
/// uniform law, stopping when a step moves less than `tol` in total variation or after `max_steps`.
/// Returns `(law, steps)`.
///
/// # Panics
///
/// If `d` is zero, `n d` exceeds [`MAX_BITS`], or the graph carries a node count of zero.
#[must_use]
pub fn stationary_current(g: &Graph, beta: f64, rule: Rule, d: usize, tol: f64, max_steps: usize) -> (Vec<f64>, usize) {
    let n = g.n;
    assert!(d >= 1 && n >= 1, "a delay is at least one tick");
    assert!(n * d <= MAX_BITS, "{n} spins x {d} frames is more than {MAX_BITS} bits");
    // State: frames f_0 (current), f_1, ..., f_{d-1}, each n bits; f_{d-1} is the frame read.
    let frames = d;
    let states = 1usize << (n * frames);
    let mask = (1usize << n) - 1;
    // Transition rows, sparse: from state s, the next current frame y has probability prod_i q_i,
    // and the new state is (y, f_0, ..., f_{d-2}).
    let mut mu = vec![1.0 / states as f64; states];
    let mut steps = 0;
    let mut row = vec![0.0f64; 1usize << n];
    while steps < max_steps {
        let mut next = vec![0.0f64; states];
        for s in 0..states {
            let mass = mu[s];
            if mass == 0.0 {
                continue;
            }
            let current = s & mask;
            let delayed = (s >> (n * (frames - 1))) & mask;
            let cur = frame_spins(current, n);
            let del = frame_spins(delayed, n);
            let ps: Vec<f64> = (0..n).map(|i| p_plus(g, beta, rule, i, cur[i], &del)).collect();
            // Product law over the next frame, built by doubling.
            row[0] = 1.0;
            let mut len = 1usize;
            for p in &ps {
                for y in 0..len {
                    let w = row[y];
                    row[y] = w * (1.0 - p);
                    row[y | len] = w * p;
                }
                len <<= 1;
            }
            let shifted = if frames > 1 { (s << n) & ((1usize << (n * frames)) - 1) } else { 0 };
            for (y, &w) in row.iter().enumerate() {
                if w != 0.0 {
                    next[shifted | y] += mass * w;
                }
            }
        }
        steps += 1;
        let moved = 0.5 * next.iter().zip(&mu).map(|(a, b)| (a - b).abs()).sum::<f64>();
        mu = next;
        if moved < tol {
            break;
        }
    }
    let mut law = vec![0.0f64; 1usize << n];
    for (s, &m) in mu.iter().enumerate() {
        law[s & mask] += m;
    }
    (law, steps)
}

/// Probability that every coupled pair agrees in sign, `Σ_x π(x) [every edge aligned]` — for two
/// spins, `P(↑↑) + P(↓↓)`, which is `1/2` under the uniform law.
#[must_use]
pub fn aligned(g: &Graph, law: &[f64]) -> f64 {
    let n = g.n;
    law.iter()
        .enumerate()
        .filter(|&(x, _)| {
            let s = frame_spins(x, n);
            (0..n).all(|i| (g.offset[i]..g.offset[i + 1]).all(|k| s[i] == s[g.nbr[k] as usize]))
        })
        .map(|(_, p)| p)
        .sum()
}

/// The settled network, by a direct solve: the law of its current frame, and how much it moves.
#[derive(Clone, Debug, PartialEq)]
pub struct Settled {
    /// Stationary law of the current frame over its `2^n` states, bit `i` set meaning spin `i` up.
    pub law: Vec<f64>,
    /// Expected number of spins whose value CHANGES on one tick, `Σ_s μ(s) Σ_i P(s_i' ≠ s_i | s)`
    /// under the stationary law `μ` of the whole last-`d`-frames chain: the motion a fabric buys per
    /// clock edge. At `d = 1` it is the sum over the kernel's own law of every site's flip
    /// probability.
    pub moves: f64,
}

/// The stationary law of the last-`d`-frames chain by the Grassmann-Taksar-Heyman elimination
/// (Grassmann, Taksar and Heyman 1985), marginalised to the current frame, with the moves per tick.
///
/// Where [`stationary_current`] iterates until the law stops moving -- and a pinned automaton at
/// large `q`, whose every flip costs `e^{-2q}`, needs a number of steps that grows as `e^{2q}` -- this
/// costs the same at every `q`. And it is exact where a plain elimination is not: GTH never subtracts,
/// so every entry of the law comes back to a RELATIVE accuracy of a few ulps however nearly
/// decomposable the chain, which is what a distance of `1e-8` from Boltzmann needs when the chain's
/// slow modes are `e^{-2q}` from one. The price is a dense matrix over `2^{n d}` states: at the cap
/// of [`MAX_BITS`] it is 134 MB, and the elimination took 0.3 s for a pair, 0.6 s for a triangle and
/// 1.2 s for four spins (Apple silicon, release), its fill-in growing with each state's `2^n`
/// successors. At `n d = 9` it is milliseconds.
///
/// # Panics
///
/// If `d` is zero, `n d` exceeds [`MAX_BITS`], the graph has no nodes, or the chain is reducible (a
/// rule whose flip probability is exactly 0 or 1 can make it so).
#[must_use]
pub fn stationary_solved(g: &Graph, beta: f64, rule: Rule, d: usize) -> Settled {
    let n = g.n;
    assert!(d >= 1 && n >= 1, "a delay is at least one tick");
    assert!(n * d <= MAX_BITS, "{n} spins x {d} frames is more than {MAX_BITS} bits");
    let m = 1usize << (n * d);
    let mask = (1usize << n) - 1;
    let mut a = vec![0.0f64; m * m];
    let mut flips = vec![0.0f64; m];
    let mut row = vec![0.0f64; 1usize << n];
    for st in 0..m {
        let cur = frame_spins(st & mask, n);
        let del = frame_spins((st >> (n * (d - 1))) & mask, n);
        let ps: Vec<f64> = (0..n).map(|i| p_plus(g, beta, rule, i, cur[i], &del)).collect();
        // A change is measured against the CURRENT value. Against the delayed one, a spin that has just
        // flipped would count as moving on every tick of the window, and at d = 1 the two agree.
        let mut changes = 0.0;
        for i in 0..n {
            changes += if cur[i] > 0 { 1.0 - ps[i] } else { ps[i] };
        }
        flips[st] = changes;
        row[0] = 1.0;
        let mut len = 1usize;
        for p in &ps {
            for y in 0..len {
                let w = row[y];
                row[y] = w * (1.0 - p);
                row[y | len] = w * p;
            }
            len <<= 1;
        }
        let shifted = if d > 1 { (st << n) & (m - 1) } else { 0 };
        for (y, &w) in row.iter().enumerate() {
            a[st * m + (shifted | y)] += w;
        }
    }
    let mu = gth(&mut a, m);
    let total: f64 = mu.iter().sum();
    let mut law = vec![0.0f64; 1usize << n];
    let mut moves = 0.0;
    for (st, w) in mu.iter().enumerate() {
        law[st & mask] += w / total;
        moves += w / total * flips[st];
    }
    Settled { law, moves }
}

/// The stationary vector of the dense row-stochastic matrix `a` (`m x m`, row `i` the law of the
/// next state from `i`), unnormalised, by GTH elimination; `a` is consumed. State 0 must be
/// recurrent. Transient states come back with zero mass, since nothing flows into them.
fn gth(a: &mut [f64], m: usize) -> Vec<f64> {
    // GTH: censor the chain onto {0..k-1} one state at a time. Only OFF-diagonal mass is ever read,
    // so no step subtracts; the diagonal is left holding whatever it holds.
    for k in (1..m).rev() {
        let out: f64 = a[k * m..k * m + k].iter().sum();
        assert!(out > 0.0, "the chain is reducible: state {k} cannot reach any lower state");
        let (top, rest) = a.split_at_mut(k * m);
        let row_k = &rest[..k];
        for i in 0..k {
            let aik = top[i * m + k] / out;
            top[i * m + k] = aik;
            if aik == 0.0 {
                continue;
            }
            for (x, &y) in top[i * m..i * m + k].iter_mut().zip(row_k) {
                *x += aik * y;
            }
        }
    }
    let mut mu = vec![0.0f64; m];
    mu[0] = 1.0;
    for k in 1..m {
        mu[k] = (0..k).map(|i| mu[i] * a[i * m + k]).sum();
    }
    mu
}

/// A coloured fabric, settled: see [`stationary_coloured`].
#[derive(Clone, Debug, PartialEq)]
pub struct ColouredSettled {
    /// Law of the current frame at a tick chosen uniformly at random -- the average over the `K`
    /// phases of the sweep -- over its `2^n` states, bit `i` set meaning spin `i` up.
    pub law: Vec<f64>,
    /// `after[c]`: the law of the current frame just after colour class `c` has moved. At `d = 1`,
    /// with every class an independent set, each is the Boltzmann law; late, they differ.
    pub after: Vec<Vec<f64>>,
    /// Expected number of spins whose value changes on one tick, averaged over the phases. Held
    /// spins never change.
    pub moves: f64,
}

/// The stationary law of a COLOURED heat-bath fabric whose reads are `d >= 1` ticks old: on tick
/// `t` the spins of colour class `t mod K` redraw from the heat-bath conditional `σ(2β f_i)` at the
/// field of the frame `d` ticks back, and every other spin HOLDS the value it has. At `d = 1` the
/// frame read is the frame held, and with every class an independent set of the graph this is
/// chromatic Gibbs, exact. A schedule, not a [`Rule`]: the moving spins are heat-bath p-bits, which
/// ignore the value they hold, but a held spin's next value IS the value it holds -- the own-state
/// dependence by which the delay reaches [`Rule::Arrhenius`] and [`Rule::Sca`].
///
/// Exact, as a chain on (the class to move next, the last `d` frames), `K 2^{n d}` states, solved
/// by GTH elimination like [`stationary_solved`]. `classes` must partition the spins; a class that
/// contains a coupled pair is allowed (one class holding every spin is the every-tick heat bath)
/// but is then not Gibbs even when read fresh.
///
/// # Panics
///
/// If `d` is zero, `classes` is not a partition of `0..n` into non-empty classes, or `K 2^{n d}`
/// exceeds `2^`[`MAX_BITS`].
#[must_use]
pub fn stationary_coloured(g: &Graph, beta: f64, classes: &[Vec<usize>], d: usize) -> ColouredSettled {
    let n = g.n;
    let k = classes.len();
    assert!(d >= 1 && n >= 1, "a delay is at least one tick");
    let mut seen = vec![0usize; n];
    for class in classes {
        assert!(!class.is_empty(), "a colour class is empty");
        for &i in class {
            assert!(i < n, "spin {i} is not in a graph of {n}");
            seen[i] += 1;
        }
    }
    assert!(seen.iter().all(|&c| c == 1), "the classes must partition the spins, each exactly once: {seen:?}");
    let frames = 1usize << (n * d);
    assert!(k * frames <= 1usize << MAX_BITS, "{k} phases x 2^({n} x {d}) states is more than 2^{MAX_BITS}");
    let m = k * frames;
    let mask = (1usize << n) - 1;
    let mut a = vec![0.0f64; m * m];
    let mut flips = vec![0.0f64; m];
    for (c, class) in classes.iter().enumerate() {
        let moving: usize = class.iter().map(|&i| 1usize << i).sum();
        let next_phase = (c + 1) % k;
        for hist in 0..frames {
            let st = c * frames + hist;
            let held = hist & mask;
            let read = frame_spins((hist >> (n * (d - 1))) & mask, n);
            let up: Vec<f64> = (0..n).map(|i| p_plus(g, beta, Rule::HeatBath, i, spin(held, i), &read)).collect();
            flips[st] = class.iter().map(|&i| if spin(held, i) > 0 { 1.0 - up[i] } else { up[i] }).sum();
            let older = if d > 1 { (hist << n) & (frames - 1) } else { 0 };
            for y in 0..=mask {
                if y & !moving != held & !moving {
                    continue;
                }
                let w: f64 = class.iter().map(|&i| if spin(y, i) > 0 { up[i] } else { 1.0 - up[i] }).product();
                a[st * m + next_phase * frames + (older | y)] += w;
            }
        }
    }
    let mu = gth(&mut a, m);
    let total: f64 = mu.iter().sum();
    let mut law = vec![0.0f64; 1usize << n];
    let mut after = vec![vec![0.0f64; 1usize << n]; k];
    let mut moves = 0.0;
    for (st, w) in mu.iter().enumerate() {
        let (phase, hist) = (st / frames, st % frames);
        law[hist & mask] += w / total;
        // Phase `p` is the class to move NEXT, so the frame it holds is the one class `p - 1` left.
        after[(phase + k - 1) % k][hist & mask] += w;
        moves += flips[st] * w / total;
    }
    for l in &mut after {
        let mass: f64 = l.iter().sum();
        l.iter_mut().for_each(|v| *v /= mass);
    }
    ColouredSettled { law, after, moves }
}

/// The first-order constant of the delayed automaton's distance from Boltzmann:
/// `TV(π_d, π_G) e^{2q} → c_d = E_G|g_A + (d − ½) g_S| / 2` as `q → ∞` (the module docs give `g_A`
/// and `g_S`). At `d = 1` it is `E_G|Φ − <Φ>| / 2`, the constant `examples/sca_exact.rs` and
/// WORKLOADS entry 10 use for [`crate::autocorr::Kernel::Sca`]. One Poisson equation over the `2^n`
/// states of the graph, whatever the delay, so it prices delays whose chain could never be built.
///
/// The `d = 1` value follows from the proved closed form; the delay term is DERIVED by a
/// first-order expansion and held to the exact chain by this module's tests, not proved.
///
/// # Panics
///
/// If `d` is zero or the graph has more than [`crate::autocorr::MAX_DENSE_SPINS`] spins, or if the
/// Poisson system is singular to floating point, which at finite `beta` it is not.
#[must_use]
pub fn sca_rate_constant(g: &Graph, beta: f64, d: usize) -> f64 {
    use crate::autocorr::{boltzmann, lu_solve, MAX_DENSE_SPINS};
    let n = g.n;
    assert!(d >= 1, "a delay is at least one tick");
    assert!(n <= MAX_DENSE_SPINS, "{n} spins is more than the {MAX_DENSE_SPINS} a dense solve takes");
    let m = 1usize << n;
    let pi = boltzmann(g, beta).expect("n is under MAX_DENSE_SPINS, which is under MAX_SPINS");
    // Solve (L + 1 pi^T) g = r: pi^T L = 0 (Boltzmann is L's reversible law), so the added rank one
    // pins <g> = <r> = 0 and makes the system nonsingular.
    let mut a = vec![0.0f64; m * m];
    let mut r = vec![0.0f64; m];
    for x in 0..m {
        let s = frame_spins(x, n);
        let phi: Vec<f64> = (0..n).map(|i| (-beta * g.field(i, &s) * f64::from(s[i])).exp()).collect();
        a[x * m..(x + 1) * m].copy_from_slice(&pi);
        for i in 0..n {
            a[x * m + x] += phi[i];
            a[x * m + (x ^ (1 << i))] -= phi[i];
        }
        let self_term: f64 = phi.iter().map(|p| p * p - 1.0).sum();
        let mut window = 0.0;
        for i in 0..n {
            for k in g.offset[i]..g.offset[i + 1] {
                let j = g.nbr[k] as usize;
                if j > i {
                    let u = (2.0 * beta * g.w[k] * f64::from(s[i]) * f64::from(s[j])).exp();
                    window += 4.0 * phi[i] * phi[j] * (1.0 - u);
                }
            }
        }
        r[x] = self_term + (d as f64 - 0.5) * window;
    }
    assert!(lu_solve(&mut a, m, &mut r, 1), "the Poisson system is singular");
    0.5 * pi.iter().zip(&r).map(|(p, v)| p * v.abs()).sum::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autocorr::{boltzmann, peretto, sca_law, total_variation};
    use crate::graph::GraphBuilder;

    fn pair(j: f64, h: f64) -> Graph {
        let mut b = GraphBuilder::new(2);
        b.couple(0, 1, j);
        b.bias(0, h);
        b.bias(1, h);
        b.build()
    }

    fn triangle() -> Graph {
        let mut b = GraphBuilder::new(3);
        b.couple(0, 1, 1.0);
        b.couple(1, 2, 0.8);
        b.couple(0, 2, -0.6);
        b.bias(0, 0.2);
        b.bias(2, -0.1);
        b.build()
    }

    /// PAI-310's triangle (lesson `every-spin-at-once`): frustrated, three colours.
    fn frustrated() -> Graph {
        let mut b = GraphBuilder::new(3);
        b.couple(0, 1, 1.0);
        b.couple(1, 2, 1.0);
        b.couple(0, 2, -1.0);
        b.bias(0, 0.2);
        b.bias(1, -0.1);
        b.bias(2, 0.05);
        b.build()
    }

    /// Four spins, every pair coupled: `J ~ N(0, 1/4)`, `h ~ N(0, 0.01)` from numpy's
    /// `default_rng(7)`, rounded to three decimals so the literals ARE the instance.
    fn sk4() -> Graph {
        let mut b = GraphBuilder::new(4);
        for &(i, j, w) in &[
            (0, 1, 0.001),
            (0, 2, 0.149),
            (0, 3, -0.137),
            (1, 2, -0.445),
            (1, 3, -0.227),
            (2, 3, -0.496),
        ] {
            b.couple(i, j, w);
        }
        for (i, h) in [0.006, 0.134, -0.049, -0.062].into_iter().enumerate() {
            b.bias(i, h);
        }
        b.build()
    }

    fn distance(g: &Graph, q: f64, d: usize) -> f64 {
        let bolt = boltzmann(g, 1.0).expect("small");
        total_variation(&stationary_solved(g, 1.0, Rule::Sca { q }, d).law, &bolt)
    }

    /// The pinning at which the delayed automaton's law comes within `eps` of Boltzmann, bisected in
    /// a bracket of half a unit either side of the first-order `ln(c_d / eps) / 2`. The bracket is
    /// asserted, so a first-order estimate that has drifted from the exact law fails here by name.
    fn pinning_for(g: &Graph, d: usize, eps: f64) -> f64 {
        let guess = 0.5 * (sca_rate_constant(g, 1.0, d) / eps).ln();
        let (mut lo, mut hi) = (guess - 0.5, guess + 0.5);
        assert!(distance(g, lo, d) > eps && distance(g, hi, d) <= eps, "q* is not within 0.5 of {guess}");
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if distance(g, mid, d) > eps {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        hi
    }

    /// **A heat-bath fabric does not feel a uniform delay.** Its new value ignores the old one, so a
    /// `d`-tick read splits the chain into `d` interleaved synchronous chains that never meet, and the
    /// current frame follows Peretto's synchronous law at every `d` -- held to `autocorr::peretto`,
    /// which knows nothing of delays. And that law is NOT Boltzmann: the delay does not repair what
    /// synchronous updating breaks.
    #[test]
    fn a_heat_bath_fabric_samples_the_same_law_at_every_delay() {
        for g in [pair(1.0, 0.3), triangle()] {
            let beta = 1.0;
            let sync = peretto(&g, beta).expect("small");
            for d in 1..=(MAX_BITS / g.n).min(4) {
                let (law, _) = stationary_current(&g, beta, Rule::HeatBath, d, 1e-15, 20_000);
                let tv = total_variation(&law, &sync);
                assert!(tv < 1e-10, "n {}, d {d}: TV {tv:e} from Peretto", g.n);
            }
            let bolt = boltzmann(&g, beta).expect("small");
            assert!(total_variation(&sync, &bolt) > 0.05, "and it is not the Boltzmann law");
        }
    }

    /// **With the paper's own-state rule, delay moves the law, toward uniform at zero field.** Two
    /// ferromagnetically coupled spins, `βJ = 1`, clocked Arrhenius flips at `p₀ = 0.2`: the
    /// probability that they agree falls monotonically with the delay toward the uniform `1/2`, and a
    /// bias field keeps it away from `1/2`, as the paper reports. The control is the heat-bath rule on
    /// the same pair, which the delay leaves exactly where it was.
    #[test]
    fn an_arrhenius_network_forgets_its_coupling_as_the_delay_grows() {
        let g = pair(1.0, 0.0);
        let rule = Rule::Arrhenius { p0: 0.2 };
        let mut last = 1.0;
        let mut series = Vec::new();
        for d in 1..=6 {
            let (law, steps) = stationary_current(&g, 1.0, rule, d, 1e-14, 200_000);
            assert!(steps < 200_000, "d {d}: power iteration must settle");
            let a = aligned(&g, &law);
            series.push(a);
            assert!(a < last, "alignment must fall with delay: d {d}, {a} after {last}");
            assert!(a > 0.5, "and stay above uniform: {a}");
            last = a;
        }
        assert!(series[0] - series[5] > 0.05, "a real fall, not rounding: {series:?}");
        // A symmetry-breaking field restores structure the delay cannot remove.
        let biased = pair(1.0, 0.5);
        let (law, _) = stationary_current(&biased, 1.0, rule, 6, 1e-14, 200_000);
        let up = law[3];
        assert!(up > 0.4, "with a field the delayed pair still sits mostly up: P(up,up) {up}");
        // Control: the same pair under the heat-bath rule does not move at all with delay.
        let (hb1, _) = stationary_current(&g, 1.0, Rule::HeatBath, 1, 1e-15, 20_000);
        let (hb6, _) = stationary_current(&g, 1.0, Rule::HeatBath, 6, 1e-15, 20_000);
        assert!(total_variation(&hb1, &hb6) < 1e-10);
    }

    /// **The direct solve is the iterated law, for every rule.** Two methods that share only the
    /// transition rule -- power iteration from the uniform law, and GTH elimination -- on a biased pair
    /// out to four ticks of delay and a triangle out to two. The pinned automaton is included at a
    /// `q` where power iteration still settles, which is the regime the direct solve is not needed in.
    #[test]
    fn the_direct_solve_is_the_iterated_law_for_every_rule() {
        let rules = [Rule::HeatBath, Rule::Arrhenius { p0: 0.2 }, Rule::Sca { q: 1.0 }];
        for (g, dmax) in [(pair(1.0, 0.3), 4), (triangle(), 2)] {
            for rule in rules {
                for d in 1..=dmax {
                    let (iterated, steps) = stationary_current(&g, 1.0, rule, d, 1e-15, 500_000);
                    assert!(steps < 500_000, "{rule:?} d {d}: power iteration must settle");
                    let solved = stationary_solved(&g, 1.0, rule, d);
                    let gap = total_variation(&iterated, &solved.law);
                    assert!(gap < 1e-10, "{rule:?} n {} d {d}: iterated vs solved TV {gap:e}", g.n);
                }
            }
        }
    }

    /// **Read one tick late, the pinned automaton IS `Kernel::Sca`.** Its law is held to
    /// `autocorr::sca_law`'s closed form by both solvers, and its moves per tick to the sum over that
    /// law of every site's flip probability, written out here from the rule rather than read from the
    /// module. At `q = 0` the law is far from Boltzmann, so the comparison can tell a half field from
    /// a full one.
    #[test]
    fn a_pinned_automaton_read_one_tick_late_has_the_sca_closed_form() {
        for g in [pair(1.0, 0.3), frustrated()] {
            let n = g.n;
            for q in [0.0, 0.5, 2.0] {
                let want = sca_law(&g, 1.0, q).expect("small");
                let (iterated, steps) = stationary_current(&g, 1.0, Rule::Sca { q }, 1, 1e-15, 200_000);
                assert!(steps < 200_000, "q {q}: power iteration must settle");
                let solved = stationary_solved(&g, 1.0, Rule::Sca { q }, 1);
                let (a, b) = (total_variation(&iterated, &want), total_variation(&solved.law, &want));
                assert!(a < 1e-10 && b < 1e-10, "n {n} q {q}: iterated {a:e}, solved {b:e} from sca_law");
                let moves: f64 = want
                    .iter()
                    .enumerate()
                    .map(|(x, p)| {
                        let s = frame_spins(x, n);
                        p * (0..n)
                            .map(|i| {
                                let arg = 0.5 * g.field(i, &s) * f64::from(s[i]) + q;
                                1.0 / (1.0 + (2.0 * arg).exp())
                            })
                            .sum::<f64>()
                    })
                    .sum();
                assert!((solved.moves - moves).abs() < 1e-12, "q {q}: moves {} vs {moves}", solved.moves);
            }
            let bolt = boltzmann(&g, 1.0).expect("small");
            assert!(total_variation(&sca_law(&g, 1.0, 0.0).expect("small"), &bolt) > 0.1, "unpinned is far off");
        }
    }

    /// **A late read costs a pinned pair `d` times the distance.** Two spins at zero field, `βJ = 1`:
    /// to first order in `e^{-2q}` the delayed automaton's distance from Boltzmann is EXACTLY `d` times
    /// the one-tick automaton's (at zero field `g_S = 2 g_A`; module docs). The exact ratio is held to
    /// `d` at `q = 5` within `1e-3`, and its deviation must shrink between `q = 3` and `q = 5` by less
    /// than `0.05`, the `e^{-4} = 0.018` a first-order law predicts. Measured: deviations `1.2e-3`,
    /// `3.4e-3`, `6.7e-3` at `q = 3` and `2.2e-5`, `6.1e-5`, `1.3e-4` at `q = 5` for `d = 2, 3, 4`,
    /// so the bound has eight times headroom -- and a rule the delay cannot reach, which pins the
    /// DELAYED own value or reads the field from the current frame, has ratio 1 and misses by `d − 1`.
    #[test]
    fn a_late_read_costs_a_pinned_pair_d_times_the_distance() {
        let g = pair(1.0, 0.0);
        let (one3, one5) = (distance(&g, 3.0, 1), distance(&g, 5.0, 1));
        for d in 2..=4 {
            let dev3 = (distance(&g, 3.0, d) / one3 - d as f64).abs();
            let dev5 = (distance(&g, 5.0, d) / one5 - d as f64).abs();
            assert!(dev5 < 1e-3, "d {d}: TV ratio misses {d} by {dev5:e} at q = 5");
            assert!(dev5 < 0.05 * dev3, "d {d}: the miss must shrink as e^-2q: {dev3:e} -> {dev5:e}");
        }
    }

    /// **The distance follows its first-order law on graphs where it is not `d` times.** At `q = 6`
    /// the exact `e^{2q} TV` is held to `sca_rate_constant` within `5e-4` relative (measured at most
    /// `5.1e-5`, the triangle at `d = 3`; the residual is `O(e^{-2q})` and falls by `e^2` per unit of
    /// `q`) on a biased pair, PAI-310's frustrated triangle and a 4-spin SK instance. At `d = 1` the
    /// constant must equal `E_G|Φ − <Φ>| / 2`, the proved one-tick constant, to rounding: a wrong
    /// weight on the delay term, anything but `d − 1/2`, fails there first. And the law is not `d c_1`
    /// here -- the triangle's is above it and the biased pair's below -- which is what decides the
    /// next test.
    #[test]
    fn the_delayed_automaton_approaches_boltzmann_at_its_first_order_rate() {
        for (g, dmax) in [(pair(1.0, 0.3), 4), (frustrated(), 3), (sk4(), 2)] {
            let n = g.n;
            let bolt = boltzmann(&g, 1.0).expect("small");
            let phi: Vec<f64> = (0..bolt.len())
                .map(|x| {
                    let s = frame_spins(x, n);
                    (0..n).map(|i| (-g.field(i, &s) * f64::from(s[i])).exp()).sum()
                })
                .collect();
            let mean: f64 = bolt.iter().zip(&phi).map(|(p, f)| p * f).sum();
            let c1: f64 = 0.5 * bolt.iter().zip(&phi).map(|(p, f)| p * (f - mean).abs()).sum::<f64>();
            let got = sca_rate_constant(&g, 1.0, 1);
            assert!((got - c1).abs() < 1e-12 * c1, "n {n}: c_1 {got} vs E|Phi - <Phi>|/2 = {c1}");
            for d in 1..=dmax {
                let c = sca_rate_constant(&g, 1.0, d);
                let scaled = distance(&g, 6.0, d) * 12f64.exp();
                assert!((scaled / c - 1.0).abs() < 5e-4, "n {n} d {d}: e^2q TV {scaled} vs c_d {c}");
            }
        }
        let ratio = |g: &Graph, d: usize| sca_rate_constant(g, 1.0, d) / sca_rate_constant(g, 1.0, 1);
        let (tri, biased) = (ratio(&frustrated(), 3), ratio(&pair(1.0, 0.3), 3));
        assert!(tri > 3.1 && biased < 2.7, "c_3/c_1: triangle {tri} (measured 3.177), biased pair {biased} (2.647)");
    }

    /// **Pin harder, or wait? It depends on the graph, and by a few percent.** A fabric whose reads
    /// are `d` ticks old can hold the law within `ε` of Boltzmann two ways: PIN harder, running the
    /// delayed automaton every tick at `q*(ε, d)`, or WAIT, running the one-tick automaton with fresh
    /// reads on every `d`-th tick at `q*(ε, 1)`. Counted in spins moved per tick, measured exactly:
    /// on the biased pair at `d = 2`, `ε = 1e-2`, pinning moves `1.113` times as many as waiting; on
    /// the frustrated triangle at `d = 3`, `ε = 1e-3`, `0.944` times, and the first-order prediction
    /// `d c_1 / c_d` is `0.9442` there. Either way the delay costs about a factor `d`.
    #[test]
    fn pinning_beats_waiting_on_a_biased_pair_and_loses_on_a_frustrated_triangle() {
        let pin_over_wait = |g: &Graph, d: usize, eps: f64| {
            let pin = stationary_solved(g, 1.0, Rule::Sca { q: pinning_for(g, d, eps) }, d).moves;
            let wait = stationary_solved(g, 1.0, Rule::Sca { q: pinning_for(g, 1, eps) }, 1).moves / d as f64;
            pin / wait
        };
        let biased = pin_over_wait(&pair(1.0, 0.3), 2, 1e-2);
        assert!(biased > 1.05, "on the biased pair pinning must move more: {biased} (measured 1.113)");
        let g = frustrated();
        let tri = pin_over_wait(&g, 3, 1e-3);
        assert!(tri < 0.97, "on the frustrated triangle waiting must move more: {tri} (measured 0.944)");
        let predicted = 3.0 * sca_rate_constant(&g, 1.0, 1) / sca_rate_constant(&g, 1.0, 3);
        assert!((tri - predicted).abs() < 0.01, "first order predicts {predicted}, exact {tri}");
    }

    fn singletons(n: usize) -> Vec<Vec<usize>> {
        (0..n).map(|i| vec![i]).collect()
    }

    /// **Read fresh, a coloured fabric is exact Gibbs.** One class per spin -- every class an
    /// independent set -- on the zero-field pair, a biased pair, both triangles and the 4-spin SK
    /// instance: at `d = 1` the phase-averaged law and the law after every class are Boltzmann to
    /// `1e-12`. And the moves per tick are the Boltzmann average of each class's flip probability,
    /// divided by the number of classes, written out here from the heat-bath rule.
    #[test]
    fn a_coloured_fabric_read_fresh_samples_boltzmann() {
        for g in [pair(1.0, 0.0), pair(1.0, 0.3), triangle(), frustrated(), sk4()] {
            let n = g.n;
            let bolt = boltzmann(&g, 1.0).expect("small");
            let s = stationary_coloured(&g, 1.0, &singletons(n), 1);
            let tv = total_variation(&s.law, &bolt);
            assert!(tv < 1e-12, "n {n}: TV {tv:e} from Boltzmann");
            for (c, law) in s.after.iter().enumerate() {
                let tv = total_variation(law, &bolt);
                assert!(tv < 1e-12, "n {n}: after class {c}, TV {tv:e}");
            }
            let moves: f64 = bolt
                .iter()
                .enumerate()
                .map(|(x, p)| {
                    let sp = frame_spins(x, n);
                    p * (0..n)
                        .map(|i| {
                            let up = 1.0 / (1.0 + (-2.0 * g.field(i, &sp)).exp());
                            if sp[i] > 0 { 1.0 - up } else { up }
                        })
                        .sum::<f64>()
                })
                .sum::<f64>()
                / n as f64;
            assert!((s.moves - moves).abs() < 1e-12, "n {n}: moves {} vs {moves}", s.moves);
        }
    }

    /// **Read late, a coloured pair keeps each spin's marginal and loses the correlation.** With
    /// classes `{0}, {1}` the spins move on alternate ticks, and at `d >= 2` each draw reads the other
    /// spin's value from before that spin's last update: from the draw `L` ticks earlier, `L = 3` at
    /// `d = 2, 3` and `L = 5` at `d = 4, 5` (always odd, at least 3). Every draw's dependence runs back
    /// along the ticks `t, t - L, t - 2L, ...`, alternating between the spins exactly as the `d = 1`
    /// sweep alternates, and the two values in any frame were drawn on CONSECUTIVE ticks, which lie on
    /// different chains unless `L = 1`. So the two spins are independent, each with its Boltzmann
    /// marginal: the law is the product of the marginals, and at zero field the pair agrees exactly
    /// half the time. Held here at zero field (0.8808 at `d = 1`, `1/2` from `d = 2` to 5) and on a
    /// biased pair, whose law, and each phase's, must be the product of its Boltzmann marginals to
    /// `1e-12`, which is `0.28` from the Boltzmann law itself -- and which is also Peretto's
    /// synchronous law for a pair, by the same interleaving.
    #[test]
    fn a_coloured_pair_read_late_is_the_product_of_its_marginals() {
        let g = pair(1.0, 0.0);
        let fresh = aligned(&g, &stationary_coloured(&g, 1.0, &singletons(2), 1).law);
        assert!((fresh - 0.880_797_077_977_882).abs() < 1e-12, "fresh: agree {fresh}");
        for d in 2..=5 {
            let late = aligned(&g, &stationary_coloured(&g, 1.0, &singletons(2), d).law);
            assert!((late - 0.5).abs() < 1e-12, "d {d}: agree {late}, not 1/2");
        }
        let g = pair(1.0, 0.3);
        let bolt = boltzmann(&g, 1.0).expect("small");
        let (m0, m1) = (bolt[1] + bolt[3], bolt[2] + bolt[3]);
        let product = [(1.0 - m0) * (1.0 - m1), m0 * (1.0 - m1), (1.0 - m0) * m1, m0 * m1];
        assert!(total_variation(&product, &bolt) > 0.25, "the product is far from the joint law");
        // The synchronous pair interleaves the same way, so its law is the same product.
        let sync = peretto(&g, 1.0).expect("small");
        assert!(total_variation(&product, &sync) < 1e-12, "Peretto's law for a pair is the product too");
        for d in 2..=5 {
            let s = stationary_coloured(&g, 1.0, &singletons(2), d);
            for law in std::iter::once(&s.law).chain(&s.after) {
                let tv = total_variation(law, &product);
                assert!(tv < 1e-12, "d {d}: TV {tv:e} from the product of the marginals");
            }
        }
    }

    /// **On PAI-310's frustrated triangle a two-tick read moves the coloured law 0.1616 from
    /// Boltzmann**, and the three phases differ (0.1952, 0.1750, 0.1891 after classes 0, 1, 2). The
    /// figures were first computed by the PAI-310 lesson bench and by a numpy build of the same chain
    /// outside this tree; here they are held to a direct simulation of the fabric, which shares only
    /// the heat-bath formula: a ring of the last `d` frames, class `t mod 3` redrawn from the oldest,
    /// the rest copied, `1e6` ticks. Its per-phase laws are within `0.01` in total variation (the
    /// noise is a few thousandths).
    #[test]
    fn a_coloured_triangle_read_late_leaves_boltzmann() {
        let g = frustrated();
        let bolt = boltzmann(&g, 1.0).expect("small");
        let s = stationary_coloured(&g, 1.0, &singletons(3), 2);
        let tv = total_variation(&s.law, &bolt);
        assert!((tv - 0.161_573).abs() < 1e-5, "d 2: TV {tv} from Boltzmann (measured 0.161573)");
        let phases: Vec<f64> = s.after.iter().map(|l| total_variation(l, &bolt)).collect();
        for (got, want) in phases.iter().zip([0.195_177, 0.174_953, 0.189_097]) {
            assert!((got - want).abs() < 1e-5, "d 2: per-phase TVs {phases:?}");
        }

        let (d, ticks) = (2usize, 1_000_000usize);
        let mut rng = crate::rng::Pcg::new(29, 3);
        let mut ring: Vec<Vec<i8>> = vec![vec![-1; 3]; d];
        let mut seen = vec![vec![0.0f64; 8]; 3];
        for t in 0..ticks + 1000 {
            let c = t % 3;
            let read = ring[d - 1].clone();
            let mut next = ring[0].clone();
            let up = 1.0 / (1.0 + (-2.0 * g.field(c, &read)).exp());
            next[c] = if rng.f64() < up { 1 } else { -1 };
            ring.rotate_right(1);
            ring[0] = next;
            if t >= 1000 {
                let x = (0..3).filter(|&i| ring[0][i] > 0).map(|i| 1usize << i).sum::<usize>();
                seen[c][x] += 1.0;
            }
        }
        for (c, counts) in seen.iter().enumerate() {
            let total: f64 = counts.iter().sum();
            let sim: Vec<f64> = counts.iter().map(|v| v / total).collect();
            let gap = total_variation(&sim, &s.after[c]);
            assert!(gap < 0.01, "after class {c}: simulation vs chain TV {gap}");
        }
    }

    /// **The control: an every-tick heat-bath fabric is delay-blind, a coloured one is not.** On a
    /// biased pair and the frustrated triangle, the every-tick law (`stationary_solved`) is the same at
    /// `d = 1, 2, 3` to `1e-10`, and a coloured schedule with ONE class holding every spin is that
    /// fabric, delay for delay. Split into one class per spin, the same p-bits read two ticks late move
    /// more than `0.1` from where they sat read fresh.
    #[test]
    fn a_coloured_fabric_feels_the_delay_that_an_every_tick_one_does_not() {
        for g in [pair(1.0, 0.3), frustrated()] {
            let n = g.n;
            let every: Vec<Vec<f64>> = (1..=3).map(|d| stationary_solved(&g, 1.0, Rule::HeatBath, d).law).collect();
            let all: Vec<usize> = (0..n).collect();
            for d in 1..=3 {
                let blind = total_variation(&every[d - 1], &every[0]);
                assert!(blind < 1e-10, "n {n}, d {d}: the every-tick law moved {blind:e}");
                let one = stationary_coloured(&g, 1.0, std::slice::from_ref(&all), d).law;
                let same = total_variation(&one, &every[d - 1]);
                assert!(same < 1e-10, "n {n}, d {d}: one class vs every tick, TV {same:e}");
            }
            let fresh = stationary_coloured(&g, 1.0, &singletons(n), 1).law;
            let late = stationary_coloured(&g, 1.0, &singletons(n), 2).law;
            let moved = total_variation(&fresh, &late);
            assert!(moved > 0.1, "n {n}: the coloured law moved only {moved} with a two-tick read");
        }
    }
}
