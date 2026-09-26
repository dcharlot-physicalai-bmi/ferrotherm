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
//!
//! The chain holds the last `d` frames (the update reads the oldest), `n d` bits, so it is exact
//! only for small networks: 12 bits, `2^12` augmented states, is the cap used here.

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autocorr::{boltzmann, peretto, total_variation};
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
}
