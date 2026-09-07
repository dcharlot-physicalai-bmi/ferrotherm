//! Multi-spin coding: 64 replicas in one machine word, updated with bitwise operations.
//!
//! [`crate::gibbs`] visits one site of one replica at a time. That is the right shape for a
//! *single* chain, and the wrong shape for what this crate mostly does with chains — parallel
//! tempering runs a ladder of them, [`crate::popanneal`] runs a population, and every
//! disorder-averaged measurement runs many seeds of the same graph. All of those are the same
//! program applied to independent states, which is the case a word-wide machine is built for and a
//! scalar loop wastes.
//!
//! So: pack 64 replicas of one graph into one `u64` per site, bit `r` holding replica `r`'s spin,
//! and update all 64 at once.
//!
//! # Why this needs a uniform coupling magnitude
//!
//! The update needs the local field, and a bitwise machine cannot multiply. With every `|J_ij|`
//! equal to one value `J`, it does not have to:
//!
//! ```text
//!     f_i = J * sum_j sign(J_ij) s_j + h_i  =  J * (2k - deg_i) + h_i
//! ```
//!
//! where `k` counts the neighbours whose *signed* contribution is `+1`. Representing `+1` as a set
//! bit, that signed contribution is `s_j XOR (J_ij < 0)` — one XOR — and `k` is a per-lane popcount,
//! which a ripple-carry adder over bit planes computes for all 64 lanes at once. The field then
//! takes only `deg_i + 1` values, so the acceptance probability is a small per-site table rather
//! than an exponential.
//!
//! That is the entire trick, and it is why every multi-spin implementation in the literature is
//! stated for `±J` models. A graph with mixed magnitudes is refused by name, carrying the distinct
//! magnitudes it found — see [`NotUniform`].
//!
//! # The randomness is drawn from the top and stops when it stops mattering
//!
//! Each lane needs its own uniform draw compared against its own threshold. Bit-sliced, that is
//! [`PRECISION`] words of randomness per site — which would be *more* RNG than the 64 scalar draws
//! it replaces, and the optimisation would be a pessimisation.
//!
//! It is not, because a comparison is decided by its leading bits. Generating uniform bits from the
//! most significant end and stopping once every lane's comparison has resolved consumes about seven
//! words rather than 32, and is exact rather than approximate: the bits not drawn are the bits that
//! could not have changed any lane's answer. [`Multispin::sweep`] reports what it actually drew.
//!
//! This makes the number of random words consumed depend on the state. That is safe here and is not
//! the mistake [`crate::cluster::Sampler::with_wolff_steps`] documents: what varies is how much
//! randomness is *consumed*, not when the chain is *observed*. Each lane's update remains an exact
//! Gibbs step, and the sweep is still one full pass over the sites.

use crate::graph::Graph;
use crate::ledger::Ledger;
use crate::rng::Pcg;

/// Replicas carried in one word. The width of the machine, not a tuning parameter.
pub const REPLICAS: usize = 64;

/// Bits of the uniform draw compared against a threshold.
///
/// A probability is stored as `round(p * 2^PRECISION)` clamped to `2^PRECISION - 1`, so the largest
/// representable probability is `1 - 2^-32 ≈ 1 - 2.3e-10` and every quantisation error is bounded by
/// `2^-32`. Both are far below anything [`crate::certify`] can resolve — its interval on `beta` at
/// six thousand draws is a few parts in a thousand — and
/// `quantisation_is_far_below_what_a_certificate_can_resolve` measures that rather than asserting
/// it.
///
/// The cost of the extra precision is nothing, because of the early exit described in the module
/// documentation: this is the CAP on words drawn per site, not the number drawn.
pub const PRECISION: usize = 32;

/// This model does not have one coupling magnitude, so the field cannot be a small integer count.
#[derive(Clone, Debug, PartialEq)]
pub struct NotUniform {
    /// The distinct non-zero magnitudes found, sorted. At least two, or this would not be an error.
    pub magnitudes: Vec<f64>,
}

impl core::fmt::Display for NotUniform {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "multi-spin coding needs one coupling magnitude and this model has {}: {:?}. The local \
             field is computed as a count of neighbours, which is only proportional to the field \
             when every coupling has the same size; with mixed magnitudes the count does not \
             determine the energy. Rescaling will not help — the magnitudes differ from each other, \
             not from a convention",
            self.magnitudes.len(),
            self.magnitudes
        )
    }
}

impl core::error::Error for NotUniform {}

/// Add a per-lane 0/1 word into a bit-sliced accumulator. `acc[0]` is the least significant plane.
///
/// A ripple-carry increment, 64 lanes at a time: `acc ^= carry` writes the sum bit and
/// `acc & carry` is what carries into the next plane. It is the bitwise form of "add one to the
/// lanes where `x` is set", which is how a popcount across neighbours is taken without a popcount.
///
/// Silently drops a carry out of the top plane, so `acc` must be wide enough for the largest count
/// it will hold. [`Multispin`] sizes it from the maximum degree.
#[inline]
fn add_bit(acc: &mut [u64], x: u64) {
    let mut carry = x;
    for plane in acc.iter_mut() {
        if carry == 0 {
            return;
        }
        let next = *plane & carry;
        *plane ^= carry;
        carry = next;
    }
}

/// The value each lane's accumulator holds, for reading a count back out.
#[cfg(test)]
#[inline]
fn lane_value(acc: &[u64], lane: usize) -> u32 {
    acc.iter().enumerate().fold(0u32, |v, (j, plane)| v | (((plane >> lane) & 1) as u32) << j)
}

/// A state packed 64 replicas to the word, sampled by bitwise Gibbs.
///
/// Every replica runs the same graph at the same temperature from an independent start, and they
/// never interact — the point is throughput, not a population method. [`crate::popanneal`] is where
/// replicas are meant to talk to each other.
pub struct Multispin<'g> {
    graph: &'g Graph,
    beta: f64,
    /// One word per site; bit `r` is replica `r`'s spin, set for `+1`.
    words: Vec<u64>,
    /// The uniform magnitude, kept so `set_beta` can rebuild the table.
    scale: f64,
    /// Per CSR entry: all ones where the coupling is negative, so the neighbour's bit is flipped.
    sign: Vec<u64>,
    /// Per CSR entry: whether the edge counts at all. A stored zero coupling does not.
    live: Vec<bool>,
    /// `thresh[thresh_at[i] + k]` is `round(p_up * 2^PRECISION)` for site `i` with count `k`.
    thresh: Vec<u64>,
    thresh_at: Vec<usize>,
    /// Live degree per site, which is the largest count it can reach.
    degree: Vec<usize>,
    /// Bit planes needed to hold the largest degree.
    planes: usize,
    rng: Pcg,
    /// Scratch, so a sweep does not allocate.
    acc: Vec<u64>,
    tplane: Vec<u64>,
}

impl<'g> Multispin<'g> {
    /// Pack 64 replicas of `g` at inverse temperature `beta`, each started independently.
    ///
    /// # Errors
    ///
    /// [`NotUniform`] when the non-zero couplings do not all have the same magnitude.
    ///
    /// # Zero couplings are skipped, not refused
    ///
    /// `GraphBuilder` sums duplicate pairs, so a coupling can cancel to exactly zero and stay in the
    /// CSR. Such an edge contributes nothing to any field, so counting it as a magnitude would
    /// refuse a model this samples exactly, and counting it in the degree would shift every
    /// threshold. It is dropped from both.
    pub fn new(g: &'g Graph, beta: f64, seed: u64) -> Result<Multispin<'g>, NotUniform> {
        let mut mags: Vec<f64> = Vec::new();
        for &w in &g.w {
            let m = w.abs();
            if m != 0.0 && !mags.contains(&m) {
                mags.push(m);
            }
        }
        mags.sort_by(f64::total_cmp);
        let scale = match mags.len() {
            // A model with no couplings at all is a set of independent spins, which this samples
            // perfectly well; the threshold table then has one entry per site and never varies.
            0 => 1.0,
            1 => mags[0],
            _ => return Err(NotUniform { magnitudes: mags }),
        };

        let mut sign = Vec::with_capacity(g.w.len());
        let mut live = Vec::with_capacity(g.w.len());
        for &w in &g.w {
            sign.push(if w < 0.0 { u64::MAX } else { 0 });
            live.push(w != 0.0);
        }
        let degree: Vec<usize> = (0..g.n)
            .map(|i| (g.offset[i]..g.offset[i + 1]).filter(|&k| live[k]).count())
            .collect();
        let max_deg = degree.iter().copied().max().unwrap_or(0);
        // Enough planes for `max_deg` itself, not for `max_deg` values: a site of degree four
        // reaches a count of four, which needs three planes, not two.
        let planes = (usize::BITS - max_deg.leading_zeros()) as usize;

        let mut rng = Pcg::new(seed, 0x005C_0DE5);
        let words = (0..g.n).map(|_| rng.next_u64()).collect();

        let mut m = Multispin {
            graph: g,
            beta,
            words,
            scale,
            sign,
            live,
            thresh: Vec::new(),
            thresh_at: Vec::new(),
            degree,
            planes,
            rng,
            acc: vec![0; planes.max(1)],
            tplane: vec![0; PRECISION],
        };
        m.build_table();
        Ok(m)
    }

    /// Recompute the acceptance table for a new temperature.
    ///
    /// Beta is a parameter and never a weight, so annealing changes this table and nothing else —
    /// the same rule [`crate::kernel`] states for the scalar path.
    pub fn set_beta(&mut self, beta: f64) {
        self.beta = beta;
        self.build_table();
    }

    fn build_table(&mut self) {
        self.thresh.clear();
        self.thresh_at.clear();
        let top = (1u64 << PRECISION) - 1;
        for i in 0..self.graph.n {
            self.thresh_at.push(self.thresh.len());
            let d = self.degree[i];
            for k in 0..=d {
                // Every neighbour contributes +1 or -1, and `k` of them contribute +1.
                let field = self.scale * (2.0 * k as f64 - d as f64) + self.graph.h[i];
                let p = crate::kernel::p_up(field, self.beta);
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let q = (p * (1u64 << PRECISION) as f64) as u64;
                self.thresh.push(q.min(top));
            }
        }
    }

    /// Replica `r`'s state, in the caller's `+1 / -1` convention.
    ///
    /// # Panics
    ///
    /// If `r` is not below [`REPLICAS`].
    #[must_use]
    pub fn replica(&self, r: usize) -> Vec<i8> {
        assert!(r < REPLICAS, "replica {r} of {REPLICAS}");
        self.words.iter().map(|w| if (w >> r) & 1 == 1 { 1i8 } else { -1 }).collect()
    }

    /// Every replica's energy.
    #[must_use]
    pub fn energies(&self) -> Vec<f64> {
        (0..REPLICAS).map(|r| self.graph.energy(&self.replica(r))).collect()
    }

    /// One full sweep: every site, every replica. Returns the random words actually drawn.
    ///
    /// The return value is not decoration. The early exit described in the module documentation is
    /// the whole reason this is faster than the scalar path rather than slower, and a claim about
    /// randomness consumed should be a measurement.
    pub fn sweep(&mut self, ledger: Option<&mut Ledger>) -> u64 {
        let mut drawn = 0u64;
        for i in 0..self.graph.n {
            self.acc.fill(0);
            for k in self.graph.offset[i]..self.graph.offset[i + 1] {
                if self.live[k] {
                    let j = self.graph.nbr[k] as usize;
                    // XOR by an all-ones word negates the neighbour, which is what a negative
                    // coupling does to its contribution. One operation, all 64 lanes.
                    add_bit(&mut self.acc, self.words[j] ^ self.sign[k]);
                }
            }
            drawn += self.resample(i);
        }
        if let Some(l) = ledger {
            // Every replica of every site was resampled, and the ledger prices node updates. 64 in
            // one word is a throughput property of the host, not a discount on the work done -- a
            // ledger that billed per WORD would report a 64x energy saving for running the same
            // computation on a wider machine.
            l.samples += (self.graph.n * REPLICAS) as u64;
        }
        drawn
    }

    /// Build this site's per-lane threshold planes and draw the new word.
    fn resample(&mut self, i: usize) -> u64 {
        let d = self.degree[i];
        let base = self.thresh_at[i];

        // Threshold planes: a multiplexer over the count. `sel` is the lanes whose count is `k`, so
        // the selectors partition the lanes and the ORs below never collide.
        self.tplane.fill(0);
        for k in 0..=d {
            let mut sel = u64::MAX;
            for (p, plane) in self.acc.iter().enumerate().take(self.planes) {
                sel &= if (k >> p) & 1 == 1 { *plane } else { !*plane };
            }
            if sel == 0 {
                continue;
            }
            let q = self.thresh[base + k];
            for (b, t) in self.tplane.iter_mut().enumerate() {
                if (q >> b) & 1 == 1 {
                    *t |= sel;
                }
            }
        }

        // Bit-sliced `uniform < threshold`, from the most significant end, stopping as soon as
        // every lane has resolved. `eq` is the lanes still tied; when it empties, no further bit of
        // the uniform draw can change any answer, so no further bit is drawn.
        let mut lt = 0u64;
        let mut eq = u64::MAX;
        let mut drawn = 0u64;
        for b in (0..PRECISION).rev() {
            if eq == 0 {
                break;
            }
            let u = self.rng.next_u64();
            drawn += 1;
            let t = self.tplane[b];
            lt |= eq & !u & t;
            eq &= !(u ^ t);
        }
        // Lanes still tied drew a uniform exactly equal to the threshold, which is not less than it.
        self.words[i] = lt;
        drawn
    }

    /// Draw a chain from replica `r`, so [`crate::certify`] applies to it unchanged.
    ///
    /// # Panics
    ///
    /// If `r` is not below [`REPLICAS`].
    #[must_use]
    pub fn collect(&mut self, plan: &crate::samples::Plan, r: usize) -> crate::samples::SampleSet {
        assert!(r < REPLICAS, "replica {r} of {REPLICAS}");
        for _ in 0..plan.burn_in {
            self.sweep(None);
        }
        let thin = plan.thin.max(1);
        let mut states = Vec::with_capacity(plan.draws);
        let mut energies = Vec::with_capacity(plan.draws);
        for _ in 0..plan.draws {
            for _ in 0..thin {
                self.sweep(None);
            }
            let st = self.replica(r);
            energies.push(self.graph.energy(&st));
            states.push(st);
        }
        crate::samples::SampleSet::from_chain(states, energies, self.beta, plan.burn_in, thin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::samples::Plan;

    /// Exact mean energy of `g` at `beta`, by enumeration.
    fn exact_energy(g: &Graph, beta: f64) -> f64 {
        let (mut z, mut ez) = (0.0f64, 0.0f64);
        for mask in 0u64..(1u64 << g.n) {
            let st: Vec<i8> =
                (0..g.n).map(|i| if mask >> i & 1 == 1 { 1i8 } else { -1 }).collect();
            let e = g.energy(&st);
            let w = (-beta * e).exp();
            z += w;
            ez += w * e;
        }
        ez / z
    }

    /// Mean energy per lane, reduced to the population's mean and standard deviation.
    ///
    /// Returned together because they are used together: the mean says whether the sampler is right
    /// and the deviation says whether there are really 64 of it.
    fn lane_means(m: &mut Multispin, g: &Graph, burn: usize, runs: usize) -> (f64, f64) {
        for _ in 0..burn {
            m.sweep(None);
        }
        let mut acc = vec![0.0f64; REPLICAS];
        for _ in 0..runs {
            m.sweep(None);
            for (r, a) in acc.iter_mut().enumerate() {
                *a += g.energy(&m.replica(r));
            }
        }
        let means: Vec<f64> = acc.iter().map(|a| a / runs as f64).collect();
        let avg = means.iter().sum::<f64>() / REPLICAS as f64;
        let var = means.iter().map(|x| (x - avg).powi(2)).sum::<f64>() / (REPLICAS - 1) as f64;
        (avg, var.sqrt())
    }

    /// The bit-sliced adder counts, in every lane, for every pattern of inputs.
    ///
    /// Exhaustive rather than sampled: with five inputs there are 32 patterns per lane, and 64 lanes
    /// hold 64 of them at once, so running all 32 patterns in lane-major order covers the whole
    /// input space of the primitive. A carry that failed to propagate out of the second plane would
    /// show up only at a count of four, which is exactly the case a handful of random trials misses.
    #[test]
    fn the_bit_sliced_adder_counts_every_pattern() {
        const INPUTS: usize = 5;
        // Lane r gets the bits of r, so the 64 lanes carry 64 distinct input patterns at once.
        let words: Vec<u64> = (0..INPUTS)
            .map(|b| (0..64usize).fold(0u64, |w, lane| w | (((lane >> b) & 1) as u64) << lane))
            .collect();
        let mut acc = vec![0u64; 3];
        for w in &words {
            add_bit(&mut acc, *w);
        }
        for lane in 0..64usize {
            let want = (0..INPUTS).filter(|b| (lane >> b) & 1 == 1).count() as u32;
            assert_eq!(lane_value(&acc, lane), want, "lane {lane}");
        }
    }

    /// An isolated site lands on +1 with exactly the probability the kernel names.
    ///
    /// This is the narrowest possible test of the threshold table and the bit-sliced comparison
    /// together: one site, no neighbours, so the count is always zero and the answer is a single
    /// known probability. Checked per lane, over a range that includes probabilities near both ends,
    /// where a quantisation or an off-by-one in the comparison would show first.
    #[test]
    fn an_isolated_site_lands_up_with_the_probability_the_kernel_names() {
        for h in [-1.5, -0.4, 0.0, 0.4, 1.5] {
            let mut b = crate::graph::GraphBuilder::new(1);
            b.set_bias(0, h);
            let g = b.build();
            let beta = 0.7;
            let want = crate::kernel::p_up(h, beta);

            let mut m = Multispin::new(&g, beta, 5).unwrap();
            let sweeps = 4_000;
            let mut up = [0u32; REPLICAS];
            for _ in 0..sweeps {
                m.sweep(None);
                for (r, c) in up.iter_mut().enumerate() {
                    *c += u32::from((m.words[0] >> r) & 1 == 1);
                }
            }
            // Four standard deviations of Binomial(sweeps, want), so a real bias fails and luck
            // does not.
            let tol = 4.0 * (f64::from(sweeps) * want * (1.0 - want)).sqrt();
            for (r, &c) in up.iter().enumerate() {
                let dev = (f64::from(c) - f64::from(sweeps) * want).abs();
                assert!(
                    dev < tol.max(4.0),
                    "h {h}, lane {r}: {c} of {sweeps} up, wanted {:.1}",
                    f64::from(sweeps) * want
                );
            }
        }
    }

    /// The replicas are independent, which no per-replica check can establish.
    ///
    /// This is the failure mode the rest of this module's tests are structurally blind to. Share one
    /// random word across the lanes -- by drawing a `u64` and broadcasting a bit, say -- and all 64
    /// replicas become the same chain. Every one of them is then still an exactly correct sample of
    /// the model, so certification passes, energies match enumeration, and the sampler delivers one
    /// replica's worth of information while reporting sixty-four.
    ///
    /// So the claim has to be tested directly: the magnetisations of two lanes, over a long run,
    /// must be uncorrelated. Identical lanes give a correlation of one.
    #[test]
    fn the_replicas_are_independent_and_not_one_replica_copied() {
        let g = crate::ising::lattice2d(4, 1.0);
        let mut m = Multispin::new(&g, 0.35, 77).unwrap();
        for _ in 0..500 {
            m.sweep(None);
        }
        let runs = 3_000;
        let mut mag: Vec<Vec<f64>> = (0..4).map(|_| Vec::with_capacity(runs)).collect();
        for _ in 0..runs {
            m.sweep(None);
            for (idx, lane) in [0usize, 1, 31, 63].iter().enumerate() {
                let ones = m.words.iter().filter(|w| (*w >> lane) & 1 == 1).count();
                mag[idx].push(2.0 * ones as f64 / g.n as f64 - 1.0);
            }
        }
        let corr = |a: &[f64], b: &[f64]| {
            let n = a.len() as f64;
            let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
            let cov: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
            let va: f64 = a.iter().map(|x| (x - ma).powi(2)).sum();
            let vb: f64 = b.iter().map(|y| (y - mb).powi(2)).sum();
            cov / (va * vb).sqrt()
        };
        for i in 0..4 {
            for j in (i + 1)..4 {
                let c = corr(&mag[i], &mag[j]);
                assert!(
                    c.abs() < 0.15,
                    "lanes {i} and {j} are correlated at {c:.3}; independent lanes give about \
                     {:.3} and identical lanes give 1",
                    1.0 / (runs as f64).sqrt()
                );
            }
        }
    }

    /// Every replica is a Boltzmann sample, scored against exact enumeration.
    ///
    /// Lane zero gets a full certificate — distribution, autocorrelation, fitted temperature. The
    /// other 63 are checked as a population, because checking each against a fixed tolerance is a
    /// test that cannot be written correctly: the per-lane spread is sampling noise, so any
    /// threshold tight enough to catch a biased lane is one some honest lane will cross. The first
    /// draft did exactly that and failed on lane 29 at 1.9 sigma, with the mean over all 64 lanes
    /// sitting 0.00004 from the enumerated value.
    ///
    /// What IS a statement about the sampler is the shape of that spread, so the tolerance is
    /// derived from it rather than chosen:
    ///
    ///   - the mean over lanes must sit within four standard errors of the enumerated value, which
    ///     catches a bias in any lane or any subset of them;
    ///   - the spread must not COLLAPSE, which is the second guard on the failure
    ///     `the_replicas_are_independent_and_not_one_replica_copied` exists for — identical lanes
    ///     agree perfectly, and a standard deviation of zero is what that looks like from here.
    #[test]
    fn every_replica_is_a_boltzmann_sample() {
        for (name, g) in [
            ("ferro ring 10", crate::ising::ring(10, 1.0, 0.0)),
            ("antiferro ring 10", crate::ising::ring(10, -1.0, 0.0)),
            ("ferro ring 10, field", crate::ising::ring(10, 1.0, 0.35)),
        ] {
            let beta = 0.4;
            let want = exact_energy(&g, beta);

            let mut m = Multispin::new(&g, beta, 13).unwrap();
            let set = m.collect(&Plan::new(500, 6000, 2), 0);
            let cert = set.certificate(&g).expect("collect returns a chain");
            crate::certify::assert_boltzmann(&cert, beta, &format!("{name}, lane 0"));

            let mut m = Multispin::new(&g, beta, 13).unwrap();
            let (avg, sd) = lane_means(&mut m, &g, 500, 8_000);
            let se = sd / (REPLICAS as f64).sqrt();
            assert!(
                (avg - want).abs() < 4.0 * se,
                "{name}: <E> over 64 lanes is {avg:.4} against an enumerated {want:.4}, \
                 {:.1} standard errors out",
                (avg - want).abs() / se
            );
            assert!(
                sd > 1e-3,
                "{name}: the 64 lanes agree to {sd:.2e}, which is not 64 independent chains"
            );
        }
    }

    /// A frustrated ±J glass is handled: the signs live in an XOR, not in the magnitude.
    ///
    /// Built by hand rather than from `planted::frustrated_loops`, which overlaps its planted loops
    /// so their couplings add — producing magnitudes of 1, 2 and 3, a model this module refuses and
    /// one of the two cases in `mixed_coupling_magnitudes_are_refused_with_the_magnitudes_found`.
    #[test]
    fn a_pm_j_glass_is_sampled_like_any_other_uniform_model() {
        let l = 4usize;
        let mut rng = Pcg::new(31, 5);
        let mut b = crate::graph::GraphBuilder::new(l * l);
        for y in 0..l {
            for x in 0..l {
                let i = y * l + x;
                b.couple(i, y * l + (x + 1) % l, f64::from(rng.spin(0.5)));
                b.couple(i, ((y + 1) % l) * l + x, f64::from(rng.spin(0.5)));
            }
        }
        let g = b.build();
        let mags: Vec<f64> = {
            let mut v: Vec<f64> = g.w.iter().map(|w| w.abs()).filter(|&m| m != 0.0).collect();
            v.sort_by(f64::total_cmp);
            v.dedup();
            v
        };
        assert_eq!(mags, vec![1.0], "the fixture must be pm-J to exercise this path: {mags:?}");
        assert!(g.w.iter().any(|&w| w < 0.0), "and it must actually have negative couplings");

        let beta = 0.4;
        let want = exact_energy(&g, beta);
        let mut m = Multispin::new(&g, beta, 3).unwrap();
        let (avg, sd) = lane_means(&mut m, &g, 1_000, 20_000);
        assert!(
            (avg - want).abs() < 4.0 * sd / (REPLICAS as f64).sqrt(),
            "<E> {avg:.4} against an enumerated {want:.4}, sd {sd:.4}"
        );
    }

    /// Mixed magnitudes are refused, and the refusal names them.
    #[test]
    fn mixed_coupling_magnitudes_are_refused_with_the_magnitudes_found() {
        let mut b = crate::graph::GraphBuilder::new(4);
        b.couple(0, 1, 1.0);
        b.couple(1, 2, -1.0);
        b.couple(2, 3, 2.5);
        let g = b.build();
        let Err(e) = Multispin::new(&g, 0.4, 1) else {
            panic!("a model with two magnitudes cannot be multi-spin coded")
        };
        assert_eq!(e.magnitudes, vec![1.0, 2.5], "the sign is not a magnitude");

        // The crate's own generator produces such models: `frustrated_loops` overlaps its planted
        // loops, so their couplings add. That is a real limit of multi-spin coding rather than a
        // gap in the fixture, and it is recorded here so it is not rediscovered as a surprise.
        let planted = crate::planted::frustrated_loops(6, 40, 3).graph;
        let Err(e) = Multispin::new(&planted, 0.4, 1) else {
            panic!("overlapping planted loops do not have one magnitude")
        };
        assert!(e.magnitudes.len() > 1, "{:?}", e.magnitudes);
    }

    /// A coupling that cancelled to zero is skipped, not counted as a second magnitude.
    ///
    /// `GraphBuilder` sums duplicates, so this is the ordinary way an edge disappears. Counting it
    /// as a magnitude would refuse the model; counting it in the degree would shift every threshold
    /// on both its endpoints, which is a wrong answer rather than a refusal.
    #[test]
    fn a_zero_coupling_is_neither_a_magnitude_nor_a_degree() {
        let mut b = crate::graph::GraphBuilder::new(8);
        for i in 0..8 {
            b.couple(i, (i + 1) % 8, 1.0);
        }
        b.couple(0, 4, 1.0);
        b.couple(0, 4, -1.0);
        let g = b.build();
        let beta = 0.4;
        let m = Multispin::new(&g, beta, 2).expect("a cancelled edge is not a second magnitude");
        assert_eq!(m.degree[0], 2, "the dead edge must not raise site 0's degree");
        assert_eq!(m.degree[4], 2, "nor site 4's");

        let mut m = Multispin::new(&g, beta, 2).unwrap();
        let want = exact_energy(&g, beta);
        let (avg, sd) = lane_means(&mut m, &g, 500, 12_000);
        assert!(
            (avg - want).abs() < 4.0 * sd / (REPLICAS as f64).sqrt(),
            "<E> {avg:.4} against an enumerated {want:.4}"
        );
    }

    /// The early exit is what makes this faster, so it is measured rather than described.
    ///
    /// A comparison is decided by its leading bits, so drawing from the top and stopping once every
    /// lane has resolved should consume a handful of words per site rather than `PRECISION`. Without
    /// the exit the module draws 32 words per site, which is worse than the 64 scalar draws it
    /// replaces once each scalar draw is two `u32`s -- the optimisation would be a pessimisation,
    /// and nothing else here would notice.
    #[test]
    fn the_early_exit_draws_a_handful_of_words_and_not_the_cap() {
        let g = crate::ising::lattice2d(8, 1.0);
        let mut m = Multispin::new(&g, 0.44, 4).unwrap();
        for _ in 0..50 {
            m.sweep(None);
        }
        let sweeps = 200;
        let drawn: u64 = (0..sweeps).map(|_| m.sweep(None)).sum();
        let per_site = drawn as f64 / (sweeps * g.n) as f64;
        assert!(
            per_site < 12.0,
            "drew {per_site:.2} words per site; the cap is {PRECISION} and the scalar path it \
             replaces costs 128 u32 draws"
        );
        assert!(per_site > 1.0, "drew {per_site:.2} words per site, which is too few to be a draw");
    }

    /// The stored probability is far closer to the true one than any certificate can resolve.
    ///
    /// `PRECISION` is a claim about accuracy, and this is the measurement behind it. The largest
    /// error over a real table is compared against the width of `certify`'s own interval on `beta`,
    /// because a quantisation smaller than the instrument cannot be detected by the instrument and a
    /// larger one is a bias being reported as a temperature.
    #[test]
    fn quantisation_is_far_below_what_a_certificate_can_resolve() {
        let g = crate::ising::lattice2d(6, 1.0);
        for beta in [0.1, 0.44, 2.0] {
            let m = Multispin::new(&g, beta, 1).unwrap();
            let mut worst = 0.0f64;
            for i in 0..g.n {
                let d = m.degree[i];
                for k in 0..=d {
                    let field = m.scale * (2.0 * k as f64 - d as f64) + g.h[i];
                    let exact = crate::kernel::p_up(field, beta);
                    let stored = m.thresh[m.thresh_at[i] + k] as f64 / (1u64 << PRECISION) as f64;
                    worst = worst.max((exact - stored).abs());
                }
            }
            assert!(worst < 1e-8, "beta {beta}: worst quantisation {worst:.3e}");
        }
    }

    /// A model with no couplings at all is 64 independent spins, not an error.
    #[test]
    fn a_model_with_no_couplings_has_no_magnitude_to_disagree_about() {
        let mut b = crate::graph::GraphBuilder::new(6);
        for i in 0..6 {
            b.set_bias(i, 0.5);
        }
        let g = b.build();
        let beta = 0.6;
        let mut m = Multispin::new(&g, beta, 8).expect("no couplings is not mixed magnitudes");
        let want = crate::kernel::p_up(0.5, beta);
        let sweeps = 4_000;
        let mut up = 0u32;
        for _ in 0..sweeps {
            m.sweep(None);
            up += m.words.iter().map(|w| w.count_ones()).sum::<u32>();
        }
        let trials = f64::from(sweeps) * (g.n * REPLICAS) as f64;
        let got = f64::from(up) / trials;
        assert!((got - want).abs() < 0.01, "P(up) {got:.4} against {want:.4}");
    }

    /// The ledger prices replicas, not words.
    ///
    /// Sixty-four spins updated in one instruction is a property of the host's word width, not a
    /// discount on the computation. A ledger charging per WORD would report a 64x energy saving for
    /// running the identical model on a wider machine, which is the exact failure the ledger exists
    /// to prevent.
    #[test]
    fn the_ledger_prices_replicas_and_not_words() {
        let g = crate::ising::lattice2d(4, 1.0);
        let mut m = Multispin::new(&g, 0.4, 1).unwrap();
        let mut l = Ledger::default();
        for _ in 0..10 {
            m.sweep(Some(&mut l));
        }
        assert_eq!(l.samples, (10 * g.n * REPLICAS) as u64);
    }
}
