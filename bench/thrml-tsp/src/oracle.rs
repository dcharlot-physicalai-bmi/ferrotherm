//! Exact energies for the scaled workloads.
//!
//! An open `L×L` Ising grid has a transfer matrix of width `2^L`. Applying it site by site
//! is `O(L² 2^L)`, which is exact through `L = 16` and is not a `2^{32}` or `2^{64}`
//! enumeration. `L = 32` and `L = 64` are past that. Their comparison is Onsager's
//! infinite-lattice energy, which is a different number from the finite lattice, plus
//! agreement of a Gibbs chain with that limit where the correlation length is short.
//!
//! Potts uses the same row transfer in base `q`. A `4×4` `q = 4` grid is `4^4 = 256`
//! states per row. An `8×8` `q = 4` grid is `65536`, still one vector.
//!
//! THRML's Ising quickstart and the spin-model notebook
//! (<https://docs.thrml.ai/en/latest/02_spin_models.html>) sample by checkerboard block
//! Gibbs. These grids are bipartite, so the blocks are that two-colouring. The notebook's
//! own large example is a Pegasus graph with normal weights, timed on eight B200s. That
//! graph is not an enumerable law. The check that can be made here is the same block
//! Gibbs on lattices whose energy is known.

use ferrotherm::ising::{exact_boltzmann, grid2d};
use ferrotherm::pfaffian;
use ferrotherm::potts::{self, Interaction, PottsBuilder};

/// `L` above this is not a transfer-matrix workload. `2^16` is the last vector that fits
/// the `O(L² 2^L)` multiply in a test.
pub const ISING_TM_MAX_SIDE: usize = 16;

#[derive(Clone, Copy, Debug)]
pub struct LatticeEnergy {
    pub log_z: f64,
    pub mean_energy: f64,
}

/// Open ferromagnetic Ising grid, `E = −J Σ_{⟨ij⟩} s_i s_j`, no field.
///
/// # Errors
///
/// `L = 0` or `L > 16`. A non-finite coupling or temperature.
pub fn ising_open_transfer(l: usize, j: f64, beta: f64) -> Result<LatticeEnergy, String> {
    if l == 0 || l > ISING_TM_MAX_SIDE {
        return Err(format!("Ising transfer matrix is exact for side 1..={ISING_TM_MAX_SIDE}, got {l}"));
    }
    if !j.is_finite() || !beta.is_finite() {
        return Err("Ising transfer matrix refuses a non-finite J or beta".into());
    }
    let k = beta * j;
    let nstate = 1usize << l;
    let mut log_w = vec![0.0; nstate];
    let mut mean_e = vec![0.0; nstate];
    for mask in 0..nstate {
        let bonds = ising_horiz(mask, l);
        log_w[mask] = k * bonds;
        mean_e[mask] = -j * bonds;
    }
    for _ in 1..l {
        let (lw, me) = ising_advance(&log_w, &mean_e, l, k, j);
        log_w = lw;
        mean_e = me;
    }
    collapse(&log_w, &mean_e)
}

fn ising_horiz(mask: usize, l: usize) -> f64 {
    let mut bonds = 0.0;
    for i in 0..l.saturating_sub(1) {
        bonds += ising_spin(mask, i) * ising_spin(mask, i + 1);
    }
    bonds
}

fn ising_spin(mask: usize, i: usize) -> f64 {
    if mask & (1 << i) == 0 { -1.0 } else { 1.0 }
}

fn ising_advance(log_w: &[f64], mean_e: &[f64], l: usize, k: f64, j: f64) -> (Vec<f64>, Vec<f64>) {
    let n = log_w.len();
    let mut cur_l = log_w.to_vec();
    let mut cur_e = mean_e.to_vec();
    for i in 0..l {
        let bit = 1usize << i;
        let mut nxt_l = vec![f64::NEG_INFINITY; n];
        let mut nxt_e = vec![0.0; n];
        for mask in 0..n {
            if !cur_l[mask].is_finite() {
                continue;
            }
            let sold = ising_spin(mask, i);
            for new_up in [false, true] {
                let snew = if new_up { 1.0 } else { -1.0 };
                let m2 = if new_up { mask | bit } else { mask & !bit };
                let mut bonds = snew * sold;
                if i > 0 {
                    bonds += snew * ising_spin(m2, i - 1);
                }
                let (nl, ne) = log_sum_mean(nxt_l[m2], nxt_e[m2], cur_l[mask] + k * bonds, cur_e[mask] - j * bonds);
                nxt_l[m2] = nl;
                nxt_e[m2] = ne;
            }
        }
        cur_l = nxt_l;
        cur_e = nxt_e;
    }
    (cur_l, cur_e)
}

/// Open Potts grid, `H = −J Σ delta(s_i, s_j)`, states `0..q-1`.
///
/// # Errors
///
/// `q < 2`, `L = 0`, or `q^L` above `2^16`. A non-finite coupling or temperature.
pub fn potts_open_transfer(l: usize, q: usize, j: f64, beta: f64) -> Result<LatticeEnergy, String> {
    if q < 2 {
        return Err(format!("Potts transfer matrix needs q >= 2, got {q}"));
    }
    if l == 0 {
        return Err("Potts transfer matrix needs a side".into());
    }
    if !j.is_finite() || !beta.is_finite() {
        return Err("Potts transfer matrix refuses a non-finite J or beta".into());
    }
    let nstate = q.checked_pow(u32::try_from(l).map_err(|_| "side does not fit u32".to_string())?)
        .filter(|n| *n <= 1 << 16)
        .ok_or_else(|| format!("Potts transfer matrix refuses q^L = {q}^{l}"))?;
    let mut log_w = vec![0.0; nstate];
    let mut mean_e = vec![0.0; nstate];
    for state in 0..nstate {
        let bonds = potts_horiz(state, l, q);
        log_w[state] = beta * j * bonds;
        mean_e[state] = -j * bonds;
    }
    for _ in 1..l {
        let (lw, me) = potts_advance(&log_w, &mean_e, l, q, j, beta);
        log_w = lw;
        mean_e = me;
    }
    collapse(&log_w, &mean_e)
}

fn potts_digit(state: usize, i: usize, q: usize) -> usize {
    (state / q.pow(u32::try_from(i).unwrap_or(0))) % q
}

fn potts_set(state: usize, i: usize, q: usize, digit: usize) -> usize {
    let p = q.pow(u32::try_from(i).unwrap_or(0));
    let old = (state / p) % q;
    state - old * p + digit * p
}

fn potts_horiz(state: usize, l: usize, q: usize) -> f64 {
    let mut bonds = 0.0;
    for i in 0..l.saturating_sub(1) {
        if potts_digit(state, i, q) == potts_digit(state, i + 1, q) {
            bonds += 1.0;
        }
    }
    bonds
}

fn potts_advance(log_w: &[f64], mean_e: &[f64], l: usize, q: usize, j: f64, beta: f64) -> (Vec<f64>, Vec<f64>) {
    let n = log_w.len();
    let mut cur_l = log_w.to_vec();
    let mut cur_e = mean_e.to_vec();
    for i in 0..l {
        let mut nxt_l = vec![f64::NEG_INFINITY; n];
        let mut nxt_e = vec![0.0; n];
        for state in 0..n {
            if !cur_l[state].is_finite() {
                continue;
            }
            let old = potts_digit(state, i, q);
            for digit in 0..q {
                let m2 = potts_set(state, i, q, digit);
                let mut bonds = if digit == old { 1.0 } else { 0.0 };
                if i > 0 && digit == potts_digit(m2, i - 1, q) {
                    bonds += 1.0;
                }
                let (nl, ne) = log_sum_mean(
                    nxt_l[m2],
                    nxt_e[m2],
                    cur_l[state] + beta * j * bonds,
                    cur_e[state] - j * bonds,
                );
                nxt_l[m2] = nl;
                nxt_e[m2] = ne;
            }
        }
        cur_l = nxt_l;
        cur_e = nxt_e;
    }
    (cur_l, cur_e)
}

fn log_sum_mean(l1: f64, e1: f64, l2: f64, e2: f64) -> (f64, f64) {
    if !l1.is_finite() {
        return (l2, e2);
    }
    if !l2.is_finite() {
        return (l1, e1);
    }
    let m = l1.max(l2);
    let w1 = (l1 - m).exp();
    let w2 = (l2 - m).exp();
    let w = w1 + w2;
    (m + w.ln(), (w1 * e1 + w2 * e2) / w)
}

fn collapse(log_w: &[f64], mean_e: &[f64]) -> Result<LatticeEnergy, String> {
    let mut acc_l = f64::NEG_INFINITY;
    let mut acc_e = 0.0;
    for (i, &lw) in log_w.iter().enumerate() {
        if lw.is_finite() {
            let (nl, ne) = log_sum_mean(acc_l, acc_e, lw, mean_e[i]);
            acc_l = nl;
            acc_e = ne;
        }
    }
    if !acc_l.is_finite() || !acc_e.is_finite() {
        return Err("transfer matrix produced a non-finite partition function".into());
    }
    Ok(LatticeEnergy { log_z: acc_l, mean_energy: acc_e })
}

/// Mean energy of a long heat-bath chain on the finite open grid.
///
/// This is a Monte Carlo reference, not a closed form. It is what 32×32 and 64×64 use
/// instead of Onsager's infinite lattice. The chain is 200 sweeps of burn-in and 400
/// kept draws, seed `seed`, started from all −1.
pub fn ising_reference_chain(side: usize, j: f64, beta: f64, seed: u64) -> f64 {
    let g = grid2d(side, side, j);
    let mut smp = ferrotherm::gibbs::Sampler::new(&g, beta, seed);
    smp.s.fill(-1);
    let set = smp.collect(&ferrotherm::samples::Plan::new(200, 400, 1), None);
    let e = set.energies();
    e.iter().sum::<f64>() / e.len() as f64
}

/// Mean energy of an open Ising grid by enumeration. `None` above 20 spins.
pub fn ising_enumerated_mean(l: usize, j: f64, beta: f64) -> Option<f64> {
    if l * l > 20 {
        return None;
    }
    let g = grid2d(l, l, j);
    let p = exact_boltzmann(&g, beta);
    let mut s = vec![-1i8; g.n];
    let mut mean = 0.0;
    for (mask, &pk) in p.iter().enumerate() {
        for b in 0..g.n {
            s[b] = if mask >> b & 1 == 1 { 1 } else { -1 };
        }
        mean += pk * g.energy(&s);
    }
    Some(mean)
}

/// Kac–Ward energy of the same open grid, for the sizes whose dart matrix is small.
pub fn ising_kac_ward_energy(l: usize, j: f64, beta: f64) -> Result<f64, String> {
    let g = grid2d(l, l, j);
    let sol = pfaffian::solve(&g, beta).map_err(|e| format!("Kac-Ward {l}x{l}: {e:?}"))?;
    sol.energy.ok_or_else(|| "Kac-Ward did not return an internal energy".into())
}

/// Every state of an open Potts grid, streamed. `3^16` is about 43 million states.
///
/// The mean is accumulated in log space. The distribution is not stored.
pub fn potts_open_brute_mean(l: usize, q: usize, j: f64, beta: f64) -> Result<f64, String> {
    if q < 2 || l == 0 {
        return Err("brute Potts needs q >= 2 and a side".into());
    }
    let n = l * l;
    let states = q.checked_pow(u32::try_from(n).map_err(|_| "n".to_string())?).ok_or("state count overflow")?;
    let mut edges = Vec::new();
    for y in 0..l {
        for x in 0..l {
            let i = y * l + x;
            if x + 1 < l {
                edges.push((i, i + 1));
            }
            if y + 1 < l {
                edges.push((i, i + l));
            }
        }
    }
    let mut state = vec![0u8; n];
    let mut log_z = f64::NEG_INFINITY;
    let mut mean = 0.0;
    for _ in 0..states {
        let mut energy = 0.0;
        for &(a, b) in &edges {
            if state[a] == state[b] {
                energy -= j;
            }
        }
        let (nl, nm) = log_sum_mean(log_z, mean, -beta * energy, energy);
        log_z = nl;
        mean = nm;
        let mut k = 0;
        while k < n {
            let next = usize::from(state[k]) + 1;
            if next < q {
                state[k] = next as u8;
                break;
            }
            state[k] = 0;
            k += 1;
        }
    }
    if !mean.is_finite() {
        return Err("brute Potts mean was not finite".into());
    }
    Ok(mean)
}

/// Enumerated Potts mean on an open grid. `None` when `q^{L²}` exceeds the library limit.
pub fn potts_enumerated_mean(l: usize, q: usize, j: f64, beta: f64) -> Option<f64> {
    let n = l * l;
    if q.checked_pow(u32::try_from(n).ok()?)? > potts::ENUMERATION_LIMIT {
        return None;
    }
    let mut b = PottsBuilder::new(q, n, Interaction::Potts);
    for y in 0..l {
        for x in 0..l {
            let i = y * l + x;
            if x + 1 < l {
                b.couple(i, i + 1, j);
            }
            if y + 1 < l {
                b.couple(i, i + l, j);
            }
        }
    }
    let m = b.build();
    potts::enumerate(&m, beta).ok().map(|e| e.mean_energy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotherm::free_energy::onsager_energy_density;
    use ferrotherm::gibbs::Sampler;
    use ferrotherm::samples::Plan;

    #[test]
    fn ising_transfer_matches_enumeration_on_4x4_and_kac_ward_on_8x8() {
        let j = 1.0;
        let beta = 0.3;
        let tm4 = ising_open_transfer(4, j, beta).unwrap();
        let en = ising_enumerated_mean(4, j, beta).unwrap();
        assert!((tm4.mean_energy - en).abs() < 1e-8, "tm {} enum {}", tm4.mean_energy, en);
        let tm2 = ising_open_transfer(2, j, beta).unwrap();
        let en2 = ising_enumerated_mean(2, j, beta).unwrap();
        assert!((tm2.mean_energy - en2).abs() < 1e-8, "2x2 tm {} enum {}", tm2.mean_energy, en2);
        let tm8 = ising_open_transfer(8, j, beta).unwrap();
        let kw = ising_kac_ward_energy(8, j, beta).unwrap();
        assert!((tm8.mean_energy - kw).abs() < 1e-6, "tm {} kac-ward {}", tm8.mean_energy, kw);
        assert!(ising_open_transfer(32, j, beta).is_err());
    }

    #[test]
    fn ising_16_is_exact_and_larger_sides_use_onsager_plus_a_chain() {
        let j = 1.0;
        let beta = 0.3;
        let tm16 = ising_open_transfer(16, j, beta).unwrap();
        let onsager = onsager_energy_density(beta, j);
        let per_site = tm16.mean_energy / 256.0;
        assert!(per_site < 0.0, "a ferromagnet below infinite temperature has negative energy");
        let gap16 = (per_site - onsager).abs();
        assert!(gap16 < 0.08, "16x16 {per_site} onsager {onsager}");
        for side in [32usize, 64] {
            let g = grid2d(side, side, j);
            let mut smp = Sampler::new(&g, beta, 11);
            smp.s.fill(-1);
            let plan = Plan::new(40, 40, 1);
            let set = smp.collect(&plan, None);
            let mean = set.energies().iter().sum::<f64>() / set.energies().len() as f64;
            let density = mean / (side * side) as f64;
            let gap = (density - onsager).abs();
            assert!(
                gap < gap16,
                "{side}x{side} chain {density} is not closer to Onsager {onsager} than the exact 16x16 gap {gap16}"
            );
            let mut smp2 = Sampler::new(&g, beta, 29);
            smp2.s.fill(-1);
            let set2 = smp2.collect(&plan, None);
            let mean2 = set2.energies().iter().sum::<f64>() / set2.energies().len() as f64;
            let agree = (mean - mean2).abs() / (side * side) as f64;
            assert!(agree < 0.05, "{side} reference chains differ by {agree} per site");
        }
    }

    #[test]
    fn potts_transfer_matches_enumeration_for_q3_and_q4() {
        let j = 1.0;
        let beta = 0.5;
        let tm = potts_open_transfer(3, 3, j, beta).unwrap();
        let en = potts_enumerated_mean(3, 3, j, beta).unwrap();
        assert!((tm.mean_energy - en).abs() < 1e-7, "q=3 3x3 tm {} enum {}", tm.mean_energy, en);
        let tm4 = potts_open_transfer(2, 4, j, beta).unwrap();
        let en4 = potts_enumerated_mean(2, 4, j, beta).unwrap();
        assert!((tm4.mean_energy - en4).abs() < 1e-8, "q=4 2x2 tm {} enum {}", tm4.mean_energy, en4);
        let big = potts_open_transfer(8, 3, j, beta).unwrap();
        assert!(big.mean_energy < 0.0, "ferromagnetic Potts energy is negative");
        assert!(potts_open_transfer(8, 5, j, beta).is_err(), "5^8 does not fit the vector");
        assert!(beta < potts::critical_beta(3));
        assert!(beta < potts::critical_beta(4));
    }

    #[test]
    fn q3_4x4_transfer_matrix_matches_brute_force_enumeration() {
        let j = 1.0;
        let beta = 0.5;
        let tm = potts_open_transfer(4, 3, j, beta).unwrap();
        let brute = potts_open_brute_mean(4, 3, j, beta).unwrap();
        assert!(
            (tm.mean_energy - brute).abs() < 1e-6,
            "transfer matrix {} brute {}",
            tm.mean_energy,
            brute
        );
    }

    fn potts_chain_workload(warmup: usize, draws: usize) -> crate::spec::Workload {
        let side = 4usize;
        let q = 3usize;
        let j = 1.0;
        let n = side * side;
        let mut edges = Vec::new();
        let mut even = Vec::new();
        let mut odd = Vec::new();
        for y in 0..side {
            for x in 0..side {
                let i = y * side + x;
                if x + 1 < side {
                    edges.push((i, i + 1, j));
                }
                if y + 1 < side {
                    edges.push((i, i + side, j));
                }
                if (x + y).is_multiple_of(2) {
                    even.push(i);
                } else {
                    odd.push(i);
                }
            }
        }
        let mean = potts_open_transfer(side, q, j, 0.5).unwrap().mean_energy;
        crate::spec::Workload {
            id: "potts_open_q3_4x4",
            kind: crate::spec::Kind::Potts,
            beta: 0.5,
            warmup,
            draws,
            thin: 1,
            trials: 1,
            seed0: 1,
            source: "test",
            n,
            q,
            biases: vec![0.0; n],
            edges,
            triples: Vec::new(),
            blocks: vec![even, odd],
            exact_energy: Some(mean),
            oracle: "transfer matrix",
        }
    }

    #[test]
    fn longer_q3_4x4_chains_cover_the_transfer_matrix() {
        let j = 1.0;
        let beta = 0.5;
        let oracle = potts_open_transfer(4, 3, j, beta).unwrap().mean_energy;
        let mut means = Vec::new();
        for seed in 0..12u64 {
            let w = potts_chain_workload(1500, 1500);
            let chain = crate::ft::run(&w, seed).unwrap();
            means.push(chain.score.mean_energy);
        }
        let n = means.len() as f64;
        let mean = means.iter().sum::<f64>() / n;
        let var = means.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let half = 2.201 * (var / n).sqrt();
        let lo = mean - half;
        let hi = mean + half;
        assert!(
            oracle >= lo && oracle <= hi,
            "q=3 4x4 oracle {oracle} outside chain interval [{lo}, {hi}] mean {mean}"
        );
    }

    #[test]
    fn an_8x8_reference_chain_lands_near_the_transfer_matrix() {
        let j = 1.0;
        let beta = 0.3;
        let tm = ising_open_transfer(8, j, beta).unwrap();
        let chain = ising_reference_chain(8, j, beta, 3);
        let gap = (chain - tm.mean_energy).abs() / 64.0;
        assert!(gap < 0.05, "reference {chain} transfer {} per-site gap {gap}", tm.mean_energy);
    }

    #[test]
    fn scaled_grids_are_the_two_colours_thrml_block_gibbs_uses() {
        for side in [4usize, 8, 16, 32, 64] {
            let g = grid2d(side, side, 1.0);
            assert_eq!(g.classes.len(), 2, "{side} a grid is 2-coloured, which is THRML's checkerboard");
            let mut colour = vec![0usize; g.n];
            for (c, class) in g.classes.iter().enumerate() {
                for &i in class {
                    colour[i as usize] = c;
                }
            }
            for i in 0..g.n {
                for k in g.offset[i]..g.offset[i + 1] {
                    let jn = g.nbr[k] as usize;
                    assert_ne!(colour[i], colour[jn], "{side} monochrome edge {i}-{jn}");
                }
            }
        }
    }
}
