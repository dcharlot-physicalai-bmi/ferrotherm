//! Perfect sampling by monotone coupling from the past (Propp & Wilson 1996).
//!
//! Every other sampler in this crate hands back a state that is *approximately* Boltzmann and asks
//! the caller to believe a burn-in. This one hands back a draw that is exactly Boltzmann — no
//! burn-in, no thinning, no autocorrelation — together with the coalescence time that says how far
//! back it had to go to get it.
//!
//! # The construction
//!
//! Fix a deterministic update `s -> Phi(s, u)`, here one systematic heat-bath sweep driven by one
//! uniform per site. Start the all-`+1` and all-`-1` states at time `-T` and run BOTH forward to
//! time 0 on the SAME uniforms. If they agree at time 0 then every state started at `-T` agrees
//! with them, because the update is monotone and those two sandwich the state space; so the value
//! at time 0 does not depend on what was there at `-T`, which is what a chain running since
//! `-infinity` would have produced. That value is an exact draw from the stationary distribution.
//!
//! If they disagree, double `T` and repeat — **reusing the uniforms already spent on `-T..0`**.
//! The reuse is the algorithm. Dropping it, or running forward from time 0 and stopping when the
//! chains meet, gives a stopping time correlated with the state and a biased answer; that is the
//! classic error this method exists to avoid.
//!
//! The uniform at `(step t, site i)` is a pure function of `(seed, t, i)`, so nothing is stored and
//! the reuse across doublings is exact by construction rather than by bookkeeping.
//!
//! # When it applies, and it is not "no negative couplings"
//!
//! Monotonicity is what makes two chains enough. A heat-bath site update is increasing in its
//! neighbours exactly when every coupling it sees is non-negative, so the model must be
//! **attractive**. Fields may take either sign — they shift the threshold without reversing it —
//! and `beta` may not be negative, because a negative one flips the update upside down.
//!
//! As in [`crate::cluster`], "attractive" is a property up to gauge: relabelling `s_i -> sigma_i
//! s_i` maps `J_ij -> sigma_i sigma_j J_ij` and leaves every energy alone, so any model whose
//! signed graph is balanced (Harary 1953) is one relabelling away from ferromagnetic.
//! [`crate::cluster::gauge`] decides that in one pass and produces the relabelling, so this module
//! samples every balanced instance and refuses the rest **by name**, with the negative-product
//! cycle that proves no gauge exists. A biased draw returned quietly would be worse than no draw.
//!
//! ```
//! use ferrotherm::{cftp, ising};
//!
//! let g = ising::ring(9, 1.0, 0.2);
//! let d = cftp::exact_draws(&g, 0.4, 7, 4).unwrap();
//! assert_eq!(d.len(), 4);
//! assert!(d.iter().all(|x| x.coalesced_at >= 1)); // finite, and reported
//! ```

use crate::graph::Graph;
use crate::kernel::p_up;
use crate::ledger::Ledger;
use crate::rng::Pcg;

/// Doubling cap on how far back a draw will reach before giving up, in sweeps.
///
/// Coalescence is almost surely finite on an attractive model, but "finite" can mean longer than a
/// caller wants to wait deep in an ordered phase, so there is a bound and it is reported.
pub const DEFAULT_MAX_STEPS: usize = 1 << 20;

/// Why no draw was produced.
#[derive(Clone, Debug, PartialEq)]
pub enum Refused {
    /// No gauge makes this model attractive, so the coupled update is not monotone.
    NotAttractive(crate::cluster::Frustrated),
    /// `beta` was negative or not finite, which reverses or destroys the monotone update.
    BadBeta {
        /// The inverse temperature asked for.
        beta: f64,
    },
    /// The two chains had not met by the cap. The bound reached, in sweeps.
    NotCoalesced {
        /// How far back the last attempt reached before the cap stopped it.
        steps: usize,
    },
}

impl core::fmt::Display for Refused {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Refused::NotAttractive(c) => write!(
                f,
                "coupling from the past needs a monotone update, which needs an attractive model: \
                 {c}"
            ),
            Refused::BadBeta { beta } => write!(
                f,
                "beta {beta} is negative or not finite: the heat-bath update is increasing in its \
                 neighbours only for beta >= 0, and a monotone coupling of two chains is worthless \
                 without that"
            ),
            Refused::NotCoalesced { steps } => write!(
                f,
                "the coupled chains had not met {steps} sweeps back, which is the cap; raise it \
                 with `Perfect::with_max_steps`, or sample at a higher temperature"
            ),
        }
    }
}

impl core::error::Error for Refused {}

/// One exact draw, and the evidence that it is one.
#[derive(Clone, Debug)]
pub struct Draw {
    /// The state, in the caller's own coordinates (any gauge is undone).
    pub state: Vec<i8>,
    /// Sweeps back from time 0 at which the two chains had coalesced. Finite by construction: this
    /// value existing is what makes `state` an exact draw rather than an approximate one.
    pub coalesced_at: usize,
    /// Attempts made, so `coalesced_at` is `2^(doublings - 1)` unless the cap truncated the last.
    pub doublings: usize,
    /// Single-site updates performed, both chains and every restarted attempt included.
    pub updates: u64,
}

/// A perfect sampler for one model at one temperature.
///
/// Construction is where a model is accepted or refused; after that every draw is exact.
pub struct Perfect<'g> {
    g: &'g Graph,
    /// The gauged model, when the caller's has negative couplings. `None` means `g` is already
    /// attractive and no relabelled copy was built.
    gauged: Option<Graph>,
    sigma: Vec<i8>,
    beta: f64,
    max_steps: usize,
}

impl core::fmt::Debug for Perfect<'_> {
    /// `Graph` is not `Debug`, so this reports the sampler's own parameters rather than the model.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Perfect {{ n: {}, beta: {}, gauged: {}, max_steps: {} }}",
            self.g.n,
            self.beta,
            self.gauged.is_some(),
            self.max_steps
        )
    }
}

impl<'g> Perfect<'g> {
    /// Accept `g` at `beta`, or refuse it.
    ///
    /// # Errors
    ///
    /// [`Refused::BadBeta`] for a negative or non-finite `beta`; [`Refused::NotAttractive`],
    /// carrying the negative-product cycle, when no gauge makes the model ferromagnetic.
    pub fn new(g: &'g Graph, beta: f64) -> Result<Perfect<'g>, Refused> {
        if !(beta >= 0.0) || !beta.is_finite() {
            return Err(Refused::BadBeta { beta });
        }
        let sigma = crate::cluster::gauge(g).map_err(Refused::NotAttractive)?;
        // An all-`+1` gauge is the identity, and rebuilding the graph to apply it would cost a CSR
        // construction to change nothing.
        let gauged =
            (!sigma.iter().all(|&x| x > 0)).then(|| crate::cluster::apply_gauge(g, &sigma));
        Ok(Perfect { g, gauged, sigma, beta, max_steps: DEFAULT_MAX_STEPS })
    }

    /// The sampler [`Perfect::new`] would have refused, for the test that measures what the
    /// refusal is worth. Not public: it returns biased draws, silently, by construction.
    #[cfg(test)]
    fn unchecked(g: &'g Graph, beta: f64) -> Perfect<'g> {
        Perfect { g, gauged: None, sigma: vec![1i8; g.n], beta, max_steps: DEFAULT_MAX_STEPS }
    }

    /// Reach at most `steps` sweeps into the past before refusing.
    ///
    /// # Panics
    ///
    /// If `steps` is zero: a draw needs at least one sweep.
    #[must_use]
    pub fn with_max_steps(mut self, steps: usize) -> Perfect<'g> {
        assert!(steps > 0, "a draw needs at least one sweep");
        self.max_steps = steps;
        self
    }

    /// The relabelling that made the model attractive: `+1` everywhere when it already was.
    #[must_use]
    pub fn gauge(&self) -> &[i8] {
        &self.sigma
    }

    /// The model actually sampled: the gauged copy when there is one.
    fn model(&self) -> &Graph {
        self.gauged.as_ref().unwrap_or(self.g)
    }

    /// A state in the caller's coordinates, from one in the sampler's. Its own inverse.
    fn unmap(&self, s: &[i8]) -> Vec<i8> {
        s.iter().zip(&self.sigma).map(|(&v, &q)| v * q).collect()
    }

    /// The uniform stream for sweep `t` before time 0. A pure function of `(seed, t)`, which is
    /// what makes a doubling reuse the earlier sweeps exactly instead of approximately.
    fn stream(seed: u64, t: usize) -> Pcg {
        Pcg::new(mix(seed ^ mix(t as u64)), t as u64)
    }

    /// One systematic heat-bath sweep of every chain, on one shared uniform per site.
    fn sweep(&self, seed: u64, t: usize, chains: &mut [Vec<i8>]) {
        let g = self.model();
        let mut rng = Self::stream(seed, t);
        for i in 0..g.n {
            // ONE draw for every chain. Two chains updated from two uniforms are two independent
            // chains, and independent chains coalesce with probability zero.
            let u = rng.f64();
            for c in &mut *chains {
                c[i] = if u < p_up(g.field(i, c), self.beta) { 1 } else { -1 };
            }
        }
    }

    /// The value at time 0 of every chain started at `-steps`, or `None` if the sandwiching chains
    /// had not met by then.
    ///
    /// Returned in the caller's coordinates. This is one CFTP attempt; [`Perfect::draw`] is the
    /// doubling loop around it.
    #[must_use]
    pub fn from_past(&self, seed: u64, steps: usize) -> Option<Vec<i8>> {
        let n = self.model().n;
        let mut chains = [vec![1i8; n], vec![-1i8; n]];
        for t in (1..=steps).rev() {
            self.sweep(seed, t, &mut chains);
        }
        (chains[0] == chains[1]).then(|| self.unmap(&chains[0]))
    }

    /// Push `start` from `-steps` to time 0 through the same uniforms [`Perfect::from_past`] uses.
    ///
    /// Once `from_past` has coalesced, this lands on its value for EVERY start — that is the
    /// theorem, and it is checkable rather than assumed.
    ///
    /// # Panics
    ///
    /// If `start` is not one spin per node.
    #[must_use]
    pub fn push_forward(&self, seed: u64, steps: usize, start: &[i8]) -> Vec<i8> {
        assert_eq!(start.len(), self.g.n, "a start state is one spin per node");
        let mut chains = [self.unmap(start)];
        for t in (1..=steps).rev() {
            self.sweep(seed, t, &mut chains);
        }
        self.unmap(&chains[0])
    }

    /// One exact draw, doubling into the past until the chains coalesce.
    ///
    /// The ledger, if given, is charged one Gibbs cycle per single-site update — both chains and
    /// every restarted attempt — and one read per node for the state handed back.
    ///
    /// # Errors
    ///
    /// [`Refused::NotCoalesced`] if the chains had not met by [`Perfect::with_max_steps`].
    pub fn draw(&self, seed: u64, ledger: Option<&mut Ledger>) -> Result<Draw, Refused> {
        let n = self.model().n;
        let (mut steps, mut total, mut doublings) = (1usize, 0usize, 0usize);
        let (state, coalesced_at) = loop {
            doublings += 1;
            total += steps;
            if let Some(s) = self.from_past(seed, steps) {
                break (s, steps);
            }
            if steps >= self.max_steps {
                return Err(Refused::NotCoalesced { steps });
            }
            steps = steps.saturating_mul(2).min(self.max_steps);
        };
        let updates = 2 * total as u64 * n as u64;
        if let Some(l) = ledger {
            l.samples += updates;
            l.reads += n as u64;
        }
        Ok(Draw { state, coalesced_at, doublings, updates })
    }

    /// `k` independent exact draws, each on its own stream.
    ///
    /// Independent, not a chain: there is nothing to thin, `tau_int` comes back at its floor of
    /// 1/2, and these are worth their raw count as an effective sample size.
    ///
    /// # Errors
    ///
    /// As [`Perfect::draw`], on the first draw that fails to coalesce.
    pub fn draws(
        &self,
        seed: u64,
        k: usize,
        mut ledger: Option<&mut Ledger>,
    ) -> Result<Vec<Draw>, Refused> {
        (0..k)
            .map(|i| self.draw(mix(seed.wrapping_add(i as u64)), ledger.as_deref_mut()))
            .collect()
    }
}

/// `k` independent exact draws from `g` at `beta`, in one call.
///
/// # Errors
///
/// As [`Perfect::new`] and [`Perfect::draw`].
pub fn exact_draws(g: &Graph, beta: f64, seed: u64, k: usize) -> Result<Vec<Draw>, Refused> {
    Perfect::new(g, beta)?.draws(seed, k, None)
}

/// `splitmix64`'s finalizer: a bijection, so distinct inputs stay distinct.
const fn mix(z: u64) -> u64 {
    let mut x = z.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;

    /// An open chain, whose treewidth of 1 makes `exact::Elimination` an oracle at any length.
    fn chain(n: usize, j: f64, h: f64) -> Graph {
        let mut b = GraphBuilder::new(n);
        for i in 0..n {
            if i + 1 < n {
                b.couple(i, i + 1, j);
            }
            b.bias(i, h);
        }
        b.build()
    }

    fn states(d: &[Draw]) -> Vec<Vec<i8>> {
        d.iter().map(|x| x.state.clone()).collect()
    }

    /// Fraction of draws with site `i` up.
    fn up_rate(d: &[Draw], i: usize) -> f64 {
        d.iter().filter(|x| x.state[i] > 0).count() as f64 / d.len() as f64
    }

    /// A frustrated model is refused by name, with a cycle the caller can check itself.
    ///
    /// The refusal is the point: the monotone coupling is silently wrong on such a model — two
    /// chains still coalesce, and the state they coalesce on is not a Boltzmann draw — so a
    /// sampler that ran anyway would return a biased answer with no symptom.
    #[test]
    fn a_frustrated_triangle_is_refused_with_a_negative_cycle() {
        let mut b = GraphBuilder::new(3);
        b.couple(0, 1, 1.0);
        b.couple(1, 2, 1.0);
        b.couple(0, 2, -1.0);
        let g = b.build();

        match Perfect::new(&g, 0.5) {
            Err(Refused::NotAttractive(c)) => {
                // Harary: balance is exactly the absence of a negative-product cycle, so the
                // witness is checkable without trusting the code that produced it.
                assert!(c.product(&g).unwrap() < 0.0, "the witness must prove itself: {c}");
            }
            other => panic!("a frustrated triangle must be refused, got {other:?}"),
        }
        assert_eq!(
            Perfect::new(&crate::ising::ring(6, 1.0, 0.0), -0.1).unwrap_err(),
            Refused::BadBeta { beta: -0.1 }
        );
    }

    /// What the refusal prevents, measured rather than asserted.
    ///
    /// A frustrated model does not announce itself by failing to coalesce: the chains still meet,
    /// and quickly, so a sampler that skipped the check would return a confident answer with no
    /// symptom at all. The answer is wrong — total variation about 0.07 from the enumerated
    /// Boltzmann distribution, many times the sampling-noise floor, with the two frustrated states
    /// coming out around 20% light. The attractive control runs the same code under the floor,
    /// which is what says the number above is the model's frustration and not this test's
    /// instrument.
    #[test]
    fn the_refusal_prevents_a_measurable_bias() {
        let beta = 0.7;
        let draws = 20_000usize;
        // certify's own floor: the TV that finite sampling of 2^3 states alone produces.
        let floor = 0.5 * (8.0 / draws as f64).sqrt();
        let measure = |g: &Graph| -> (f64, usize) {
            let d = Perfect::unchecked(g, beta).draws(4, draws, None).unwrap();
            let mut hist = vec![0.0f64; 8];
            for x in &d {
                let mut k = 0usize;
                for (i, &v) in x.state.iter().enumerate() {
                    if v > 0 {
                        k |= 1 << i;
                    }
                }
                hist[k] += 1.0 / draws as f64;
            }
            let worst = d.iter().map(|x| x.coalesced_at).max().unwrap_or(0);
            (crate::ising::tv(&hist, &crate::ising::exact_boltzmann(g, beta)), worst)
        };

        let mut b = GraphBuilder::new(3);
        b.couple(0, 1, 1.0);
        b.couple(1, 2, 1.0);
        b.couple(0, 2, -1.0);
        let (tv, worst) = measure(&b.build());
        assert!(worst <= 256, "the failure is silent: it coalesced in {worst} sweeps every time");
        // Measured 0.069 against a 0.010 floor, so the bar is set at five and the margin is real.
        assert!(tv > 5.0 * floor, "frustrated: tv {tv:.4} against a {floor:.4} floor");

        let mut b = GraphBuilder::new(3);
        b.couple(0, 1, 1.0);
        b.couple(1, 2, 1.0);
        b.couple(0, 2, 1.0);
        let (tv, _) = measure(&b.build());
        assert!(tv < floor, "attractive control: tv {tv:.4} against a {floor:.4} floor");
    }

    /// Uncoupled spins: the exact answer is a product of logistics, and one sweep must suffice.
    ///
    /// The closed form is `P(s_i = +1) = sigma(2 beta h_i)`. With no neighbours the two chains see
    /// identical fields, so they coalesce in exactly one sweep — a coalescence time that came back
    /// as anything else would mean the chains were not sharing their uniforms.
    #[test]
    fn independent_spins_match_the_closed_form_and_coalesce_in_one_sweep() {
        let hs = [-0.7, -0.3, 0.1, 0.4, 0.9, 1.3];
        let mut b = GraphBuilder::new(hs.len());
        for (i, &h) in hs.iter().enumerate() {
            b.bias(i, h);
        }
        let g = b.build();
        let beta = 0.5;
        let n = 4000;
        let d = exact_draws(&g, beta, 11, n).unwrap();

        assert!(d.iter().all(|x| x.coalesced_at == 1), "uncoupled chains coalesce in one sweep");
        for (i, &h) in hs.iter().enumerate() {
            let want = p_up(h, beta);
            let got = up_rate(&d, i);
            let se = (want * (1.0 - want) / n as f64).sqrt();
            assert!(
                (got - want).abs() < 4.5 * se,
                "site {i}: {got:.4} against the closed form {want:.4}, {:.1} standard errors",
                (got - want).abs() / se
            );
        }
    }

    /// The draws are the exact Boltzmann distribution of an enumerable model.
    ///
    /// Judged by `certify`, which compares against `ising::exact_boltzmann` over the whole state
    /// space and against the total-variation distance finite sampling alone produces. Independent
    /// draws also have to pass the autocorrelation and drift findings, which they do trivially —
    /// and that is worth asserting, because it is the property no burn-in buys.
    #[test]
    fn an_enumerable_ferromagnet_is_exactly_boltzmann() {
        let g = crate::ising::ring(9, 1.0, 0.35);
        let beta = 0.4;
        let d = exact_draws(&g, beta, 2024, 8000).unwrap();
        let s = states(&d);
        let trace: Vec<f64> = s.iter().map(|x| g.energy(x)).collect();
        let cert = crate::certify::certify(&g, beta, &s, &trace);
        assert!(cert.tau_int < 1.0, "independent draws, so tau_int is 1/2: {cert}");
        crate::certify::assert_boltzmann(&cert, beta, "cftp on a 9-ring with a field");
    }

    /// A relabelled ferromagnet — mixed signs, still balanced — is sampled exactly.
    ///
    /// This is the gauge round trip end to end. The model handed in has negative couplings and
    /// looks like a spin glass; the sampler works in the relabelled coordinates and has to undo the
    /// relabelling on the way out. Forgetting that undo leaves a distribution that is Boltzmann for
    /// a DIFFERENT model, which is exactly what `certify` measures against enumeration.
    #[test]
    fn a_relabelled_ferromagnet_is_sampled_exactly() {
        let base = crate::ising::ring(9, 1.0, 0.3);
        let mut r = Pcg::new(5, 5);
        let sigma: Vec<i8> = (0..base.n).map(|_| r.spin(0.5)).collect();
        let g = crate::cluster::apply_gauge(&base, &sigma);
        assert!(
            (0..g.n).any(|i| (g.offset[i]..g.offset[i + 1]).any(|k| g.w[k] < 0.0)),
            "the fixture must actually have negative couplings"
        );

        let beta = 0.4;
        let p = Perfect::new(&g, beta).unwrap();
        assert_ne!(p.gauge(), vec![1i8; g.n], "and the sampler must actually need a gauge");
        let d = p.draws(31, 8000, None).unwrap();
        let s = states(&d);
        let trace: Vec<f64> = s.iter().map(|x| g.energy(x)).collect();
        let cert = crate::certify::certify(&g, beta, &s, &trace);
        crate::certify::assert_boltzmann(&cert, beta, "cftp on a gauged 9-ring");
    }

    /// Every start at `-T` lands on the coalesced state. This is the theorem, checked.
    ///
    /// Monotone CFTP runs two chains and claims the answer for all `2^n`. The claim rests on the
    /// update being monotone and the uniforms being shared; break either and some third start
    /// arrives somewhere else. Random starts are the test a pair of extremes cannot be.
    #[test]
    fn every_start_lands_on_the_coalesced_state() {
        let g = crate::ising::grid2d(4, 3, 0.8);
        let p = Perfect::new(&g, 0.35).unwrap();
        let mut r = Pcg::new(77, 2);
        for seed in 0..12u64 {
            let d = p.draw(seed, None).unwrap();
            for _ in 0..8 {
                let start: Vec<i8> = (0..g.n).map(|_| r.spin(0.5)).collect();
                assert_eq!(
                    p.push_forward(seed, d.coalesced_at, &start),
                    d.state,
                    "seed {seed}: a start at -{} reached a different time 0",
                    d.coalesced_at
                );
            }
        }
    }

    /// The reported coalescence time is the first in the schedule that works, not decoration.
    ///
    /// `coalesced_at` is only evidence if it is the actual meeting time: the attempt at that depth
    /// must succeed and the attempt one doubling shallower must fail. A constant, or the cap, would
    /// pass a test that only looked for a positive number.
    #[test]
    fn the_reported_coalescence_time_is_the_first_one_that_works() {
        let g = crate::ising::ring(12, 1.0, 0.1);
        let p = Perfect::new(&g, 0.45).unwrap();
        let mut deeper = 0usize;
        for seed in 0..40u64 {
            let d = p.draw(seed, None).unwrap();
            assert!(d.coalesced_at.is_power_of_two(), "the schedule doubles from one");
            assert_eq!(d.coalesced_at, 1 << (d.doublings - 1));
            assert_eq!(p.from_past(seed, d.coalesced_at).as_ref(), Some(&d.state));
            if d.coalesced_at > 1 {
                deeper += 1;
                assert!(
                    p.from_past(seed, d.coalesced_at / 2).is_none(),
                    "seed {seed} reported {} but had already coalesced at {}",
                    d.coalesced_at,
                    d.coalesced_at / 2
                );
            }
            // Going further back cannot change the answer: that invariance IS the exactness.
            assert_eq!(p.from_past(seed, d.coalesced_at * 2).as_ref(), Some(&d.state));
        }
        assert!(deeper > 0, "the fixture must sometimes need more than one sweep");
    }

    /// At infinite temperature one sweep always suffices and the draw is uniform.
    ///
    /// `beta = 0` makes every site a fair coin whatever its field, so both chains take the same
    /// value at every site of the first sweep. The closed form is exact and needs no enumeration.
    #[test]
    fn at_infinite_temperature_one_sweep_suffices_and_the_draw_is_uniform() {
        let g = crate::ising::ring(8, 1.0, 0.5);
        let n = 4000;
        let d = exact_draws(&g, 0.0, 3, n).unwrap();
        assert!(d.iter().all(|x| x.coalesced_at == 1 && x.doublings == 1));
        let se = (0.25 / n as f64).sqrt();
        for i in 0..g.n {
            assert!(
                (up_rate(&d, i) - 0.5).abs() < 4.5 * se,
                "site {i} was up {:.4} of the time at beta 0",
                up_rate(&d, i)
            );
        }
    }

    /// Marginals match exact variable elimination on a model too large to enumerate.
    ///
    /// 24 spins is 16.7 million states, past what `certify` will enumerate, but a chain has
    /// treewidth 1 and `exact::Elimination` answers it in closed form. So this is a genuine
    /// oracle on a model the distributional test cannot reach — and it is an INDEPENDENT one,
    /// computed by a different algorithm in a different module.
    #[test]
    fn marginals_match_exact_elimination_beyond_enumeration() {
        let g = chain(24, 0.6, 0.2);
        let beta = 0.35;
        let want = crate::exact::Elimination::default().marginals(&g, beta).unwrap();
        let n = 3000;
        let d = exact_draws(&g, beta, 909, n).unwrap();
        for i in 0..g.n {
            let got = up_rate(&d, i);
            let se = (want[i] * (1.0 - want[i]) / n as f64).sqrt();
            assert!(
                (got - want[i]).abs() < 4.5 * se,
                "site {i}: {got:.4} against the exact {:.4}, {:.1} standard errors",
                want[i],
                (got - want[i]).abs() / se
            );
        }
    }

    /// A cap refuses, naming the depth it reached, rather than running forever.
    #[test]
    fn a_cap_refuses_rather_than_running_forever() {
        let g = crate::ising::lattice2d(6, 1.0);
        let p = Perfect::new(&g, 1.0).unwrap().with_max_steps(4);
        assert_eq!(p.draw(1, None).unwrap_err(), Refused::NotCoalesced { steps: 4 });
        // The same model coalesces above the critical temperature, so the cap is what refused it
        // and not the model being unsamplable.
        assert!(Perfect::new(&g, 0.1).unwrap().draw(1, None).is_ok());
    }

    /// Same seed, same draw; different seeds, different draws.
    #[test]
    fn draws_are_reproducible_and_not_all_the_same() {
        let g = crate::ising::ring(10, 1.0, 0.2);
        let a = exact_draws(&g, 0.4, 12345, 64).unwrap();
        let b = exact_draws(&g, 0.4, 12345, 64).unwrap();
        assert_eq!(states(&a), states(&b));
        let mut distinct: Vec<Vec<i8>> = states(&a);
        distinct.sort_unstable();
        distinct.dedup();
        assert!(distinct.len() > 8, "64 draws collapsed to {} states", distinct.len());
    }

    /// The ledger is charged for both chains and every restarted attempt, plus the readback.
    ///
    /// Restarting from scratch at each doubling costs `1 + 2 + ... + T = 2T - 1` sweeps, and the
    /// bill has to say so: a ledger that counted only the successful attempt would under-report the
    /// work by about half, which on a device whose read dominates is the wrong half to guess at.
    #[test]
    fn the_ledger_is_charged_for_both_chains_and_the_readback() {
        let g = crate::ising::ring(10, 1.0, 0.2);
        let p = Perfect::new(&g, 0.4).unwrap();
        let mut led = Ledger::default();
        let d = p.draw(4, Some(&mut led)).unwrap();
        assert!(d.coalesced_at > 1, "the fixture must restart at least once for this to bite");
        let sweeps = 2 * d.coalesced_at - 1;
        assert_eq!(led.samples, 2 * sweeps as u64 * g.n as u64);
        assert_eq!(led.samples, d.updates);
        assert_eq!(led.reads, g.n as u64);
    }
}
