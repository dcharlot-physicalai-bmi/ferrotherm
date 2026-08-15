//! Does the accuracy floor of parallel width get worse as the action space grows?
//!
//! `examples/sequential_depth.rs` measured, on a scalar plant, that sampling width alone stalls
//! around 21% above the true optimum however many rollouts are thrown at it, and that sequential
//! refinement clears it by more than an order of magnitude. A scalar plant is a model system, and
//! the question that decides whether the result matters to robots is whether it survives dimension:
//! a 7-DOF arm samples in a 7-dimensional action space, not a 1-dimensional one.
//!
//! # The prediction, stated before the measurement
//!
//! **Predicted:** the width floor RISES with action dimension. Random sampling covers a volume, and
//! volume grows exponentially in dimension, so a fixed rollout budget explores an n-dimensional
//! space exponentially worse. Depth should still clear it, because a refinement pass moves the
//! whole mean rather than hoping a sample lands well.
//!
//! **The alternative, which would falsify that:** the floor is dimension-independent, sitting near
//! 21% at every n. That would mean the effect is a property of the algorithm's weighting rather
//! than of the geometry it samples, and the robot-scale argument would not follow.
//!
//! Either result is worth having. The prediction is recorded here so the reading afterwards is not
//! a choice.
//!
//! # Why this can be scored against truth
//!
//! For a linear system with quadratic cost the optimal cost-to-go is exact: the discrete algebraic
//! Riccati equation has a matrix solution `P`, and the optimum from `x0` is `x0ᵀ P x0`. This probe
//! iterates the DARE to convergence, so every number below is an excess over the true optimum
//! rather than a comparison against another heuristic.
//!
//! ```text
//! cargo run --release --example depth_vs_dimension
//! ```

use ferrotherm::rng::Pcg;

/// Box-Muller, matching `src/mppi.rs` so the two probes draw from the same noise.
fn gauss(rng: &mut Pcg) -> f64 {
    let u = rng.f64().max(1e-15);
    let v = rng.f64();
    (-2.0 * u.ln()).sqrt() * (core::f64::consts::TAU * v).cos()
}

// ---- small dense linear algebra, enough for n <= 8 ------------------------------------------
fn mul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut o = vec![0.0; n * n];
    for i in 0..n { for k in 0..n { let aik = a[i*n+k];
        if aik != 0.0 { for j in 0..n { o[i*n+j] += aik * b[k*n+j]; } } } }
    o
}
fn t(a: &[f64], n: usize) -> Vec<f64> {
    let mut o = vec![0.0; n*n];
    for i in 0..n { for j in 0..n { o[j*n+i] = a[i*n+j]; } }
    o
}
fn add(a: &[f64], b: &[f64]) -> Vec<f64> { a.iter().zip(b).map(|(x,y)| x+y).collect() }
fn sub(a: &[f64], b: &[f64]) -> Vec<f64> { a.iter().zip(b).map(|(x,y)| x-y).collect() }
fn eye(n: usize, s: f64) -> Vec<f64> { let mut o = vec![0.0; n*n]; for i in 0..n { o[i*n+i] = s; } o }

/// Gauss-Jordan with partial pivoting. n is small and the matrices here are well conditioned.
fn inv(a: &[f64], n: usize) -> Vec<f64> {
    let mut m = a.to_vec();
    let mut o = eye(n, 1.0);
    for c in 0..n {
        let mut piv = c;
        for r in c+1..n { if m[r*n+c].abs() > m[piv*n+c].abs() { piv = r; } }
        if piv != c { for j in 0..n { m.swap(c*n+j, piv*n+j); o.swap(c*n+j, piv*n+j); } }
        let d = m[c*n+c];
        for j in 0..n { m[c*n+j] /= d; o[c*n+j] /= d; }
        for r in 0..n { if r != c {
            let f = m[r*n+c];
            if f != 0.0 { for j in 0..n { m[r*n+j] -= f*m[c*n+j]; o[r*n+j] -= f*o[c*n+j]; } } } }
    }
    o
}
fn matvec(a: &[f64], x: &[f64], n: usize) -> Vec<f64> {
    (0..n).map(|i| (0..n).map(|j| a[i*n+j]*x[j]).sum()).collect()
}
fn quad(x: &[f64], m: &[f64], n: usize) -> f64 {
    let mx = matvec(m, x, n);
    x.iter().zip(&mx).map(|(a,b)| a*b).sum()
}

// ---- the plant -----------------------------------------------------------------------------
/// A coupled, stable plant that is comparable across dimension: each state feeds the next, so it
/// is genuinely n-dimensional rather than n independent scalar problems wearing one name.
/// Spectral radius stays below 1 (the circulant eigenvalues are 0.85 + 0.1·ω, so |λ| <= 0.95).
struct Plant { n: usize, a: Vec<f64>, b: Vec<f64>, q: Vec<f64>, r: Vec<f64> }
impl Plant {
    fn new(n: usize) -> Self {
        let mut a = vec![0.0; n*n];
        for i in 0..n { a[i*n+i] = 0.85; a[i*n + (i+1)%n] = 0.10; }
        Plant { n, a, b: eye(n,1.0), q: eye(n,1.0), r: eye(n,0.5) }
    }
    /// Iterate the DARE to a fixed point: P = Q + AᵀPA − AᵀPB(R + BᵀPB)⁻¹BᵀPA.
    fn riccati(&self) -> Vec<f64> {
        let n = self.n;
        let (at, bt) = (t(&self.a, n), t(&self.b, n));
        let mut p = self.q.clone();
        for _ in 0..20_000 {
            let atp = mul(&at, &p, n);
            let atpa = mul(&atp, &self.a, n);
            let atpb = mul(&atp, &self.b, n);
            let btpb = mul(&mul(&bt, &p, n), &self.b, n);
            let k = inv(&add(&self.r, &btpb), n);
            let corr = mul(&mul(&atpb, &k, n), &t(&atpb, n), n);
            let next = add(&self.q, &sub(&atpa, &corr));
            let d: f64 = sub(&next, &p).iter().map(|v| v.abs()).sum();
            p = next;
            if d < 1e-12 { break; }
        }
        p
    }
    fn step(&self, x: &[f64], u: &[f64]) -> Vec<f64> {
        let ax = matvec(&self.a, x, self.n);
        let bu = matvec(&self.b, u, self.n);
        add(&ax, &bu)
    }
    fn stage(&self, x: &[f64], u: &[f64]) -> f64 { quad(x, &self.q, self.n) + quad(u, &self.r, self.n) }
}

// ---- MPPI in n dimensions -------------------------------------------------------------------
fn mppi_cost(p: &Plant, rollouts: usize, passes: usize, horizon: usize,
             sigma: f64, lambda: f64, steps: usize, seed: u64) -> f64 {
    let n = p.n;
    let mut rng = Pcg::new(seed, 0);
    let mut nominal = vec![0.0f64; horizon * n];
    let mut x: Vec<f64> = vec![1.0; n];
    let mut total = 0.0;
    for _ in 0..steps {
        for _ in 0..passes {                       // SEQUENTIAL: each pass refines the last mean
            let mut costs = vec![0.0f64; rollouts];
            let mut noise = vec![0.0f64; rollouts * horizon * n];
            for r in 0..rollouts {                 // PARALLEL: rollouts are independent
                let mut xs = x.clone();
                for h in 0..horizon {
                    let mut u = vec![0.0; n];
                    for d in 0..n {
                        let z = gauss(&mut rng) * sigma;
                        noise[(r*horizon + h)*n + d] = z;
                        u[d] = nominal[h*n + d] + z;
                    }
                    costs[r] += p.stage(&xs, &u);
                    xs = p.step(&xs, &u);
                }
            }
            let best = costs.iter().cloned().fold(f64::INFINITY, f64::min);
            let w: Vec<f64> = costs.iter().map(|c| (-(c - best)/lambda).exp()).collect();
            let sw: f64 = w.iter().sum();
            for h in 0..horizon { for d in 0..n {
                let mut acc = 0.0;
                for r in 0..rollouts { acc += w[r] * noise[(r*horizon + h)*n + d]; }
                nominal[h*n + d] += acc / sw;
            } }
        }
        let u0: Vec<f64> = (0..n).map(|d| nominal[d]).collect();
        total += p.stage(&x, &u0);
        x = p.step(&x, &u0);
        for h in 0..horizon-1 { for d in 0..n { nominal[h*n+d] = nominal[(h+1)*n+d]; } }
        for d in 0..n { nominal[(horizon-1)*n + d] = 0.0; }
    }
    total
}

fn main() {
    println!("PREDICTED before measuring: the width floor RISES with dimension.");
    println!("Falsified if the floor sits near 21% at every n.\n");
    let seeds: Vec<u64> = (1..=5).collect();
    println!("{:>4} {:>10} {:>8} {:>6} {:>10} {:>9}", "n", "rollouts", "passes", "", "excess", "spread");
    let mut floors: Vec<(usize, f64)> = Vec::new();
    for &n in &[1usize, 2, 4, 8] {
        let p = Plant::new(n);
        let pm = p.riccati();
        let x0 = vec![1.0; n];
        let opt = quad(&x0, &pm, n);
        println!("\n  n = {n}   exact optimum (matrix Riccati) = {opt:.4}");
        let mut floor_here = f64::INFINITY;
        for &(k, ps) in &[(200usize,1usize),(800,1),(3200,1),(3200,4),(3200,8)] {
            let mut v: Vec<f64> = seeds.iter()
                .map(|&s| mppi_cost(&p, k, ps, 5, 0.4, 0.6, 60, s)).collect();
            v.sort_by(|a,b| a.partial_cmp(b).unwrap());
            let med = v[v.len()/2];
            let ex = (med - opt)/opt*100.0;
            let sp = (v[v.len()-1] - v[0])/opt*100.0;
            if ps == 1 { floor_here = floor_here.min(ex); }
            println!("{:>4} {:>10} {:>8} {:>6} {:>9.2}% {:>8.2}%", n, k, ps, "", ex, sp);
        }
        floors.push((n, floor_here));
    }
    println!("\n=== the floor: best accuracy reachable at ONE pass, at any width tried ===");
    for (n, f) in &floors { println!("  n = {n:>2}   floor = {f:>7.2}%"); }
    let rise = floors.last().unwrap().1 / floors[0].1;
    println!("\n  floor at n=8 over floor at n=1: {rise:.2}x");
    confound_check();
    println!("  PREDICTION {} ", if rise > 1.3 { "CONFIRMED: the floor rises with dimension" }
                                 else if rise < 0.77 { "INVERTED: the floor FALLS with dimension" }
                                 else { "FALSIFIED: the floor is roughly dimension-independent" });
}

// ---- the confound check ---------------------------------------------------------------------
// `sigma` above is PER DIMENSION, so the total perturbation ‖u‖ grows as sqrt(n). A floor that
// rises with n might therefore be an artifact of over-perturbing a larger space rather than a
// statement about the space. Re-measure n=8 with sigma scaled down by sqrt(n), and with a sweep
// either side of it, before attributing anything to dimension.
fn confound_check() {
    let n = 8;
    let p = Plant::new(n);
    let opt = quad(&vec![1.0; n], &p.riccati(), n);
    let seeds: Vec<u64> = (1..=5).collect();
    println!("\n=== confound: is the n=8 floor about DIMENSION or about SIGMA? ===");
    println!("  (sigma is per-dimension, so fixed sigma means ||u|| grows as sqrt(n) = 2.83)");
    println!("{:>8} {:>10} {:>8} {:>10} {:>9}", "sigma", "rollouts", "passes", "excess", "spread");
    for &sg in &[0.4f64, 0.4/2.828, 0.1, 0.05] {
        for &(k, ps) in &[(3200usize, 1usize), (3200, 8)] {
            let mut v: Vec<f64> = seeds.iter()
                .map(|&s| mppi_cost(&p, k, ps, 5, sg, 0.6, 60, s)).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = v[v.len()/2];
            println!("{:>8.3} {:>10} {:>8} {:>9.2}% {:>8.2}%", sg, k, ps,
                     (med-opt)/opt*100.0, (v[v.len()-1]-v[0])/opt*100.0);
        }
    }
}
