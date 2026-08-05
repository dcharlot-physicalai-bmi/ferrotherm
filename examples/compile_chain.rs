// The compilation error bound, verified EXACTLY — and the context-matching effect measured.
//
// A 3-stage stochastic program on 4 bits is compiled stage by stage onto device-native kernels
// (Z1-patch topology, hidden units marginalized). Everything is enumerable, so the bound
// (arXiv:2608.01615 Eq. 17, the chain rule of KL — the same shape as our chain law G ~ p^(D-1)):
//
//     KL(readout)  <=  KL(trajectory)  =  sum_l  E_{x ~ mu_l}[ eps_l(x) ]
//
// is checked with exact numbers, not estimates: exact target marginals, exact compiled marginals,
// exact per-factor eps. Second measurement: compile one stage under the WRONG input distribution
// (uniform) vs the true program marginal — the realized eps under the true inputs must suffer,
// which is why context matching is a refinement and not a nicety.
//
// run: cargo run --release --example compile_chain

use ferrotherm::compile::{factor_eps, fit, patch_kernel, Cpt};

const NB: usize = 4; // bits per stage
const NS: usize = 16; // states per stage

/// Stage targets: structured noisy logic, different per stage, all full-support.
fn stage_cpt(stage: usize) -> Cpt {
    let mut cpt = vec![vec![0.0; NS]; NS];
    for xm in 0..NS {
        for ym in 0..NS {
            // deterministic core: stage 0 = rotate-left; 1 = xor with parity; 2 = increment
            let core = match stage {
                0 => ((xm << 1) | (xm >> (NB - 1))) & (NS - 1),
                1 => xm ^ (if (xm.count_ones() % 2) == 1 { 0b1001 } else { 0b0011 }),
                _ => (xm + 1) & (NS - 1),
            };
            // smeared with Hamming-distance noise
            let d = ((ym ^ core) as u32).count_ones();
            cpt[xm][ym] = (0.12f64).powi(d as i32);
        }
        let z: f64 = cpt[xm].iter().sum();
        for v in cpt[xm].iter_mut() {
            *v /= z;
        }
    }
    cpt
}

/// Exact push-forward of a distribution through a CPT.
fn push(mu: &[f64], cpt: &Cpt) -> Vec<f64> {
    let mut out = vec![0.0; NS];
    for xm in 0..NS {
        if mu[xm] > 0.0 {
            for ym in 0..NS {
                out[ym] += mu[xm] * cpt[xm][ym];
            }
        }
    }
    out
}

/// Exact push-forward through a compiled kernel.
fn push_kernel(mu: &[f64], k: &ferrotherm::compile::Kernel) -> Vec<f64> {
    let mut out = vec![0.0; NS];
    for xm in 0..NS {
        if mu[xm] > 0.0 {
            let x: Vec<i8> = (0..NB).map(|b| if xm >> b & 1 == 1 { 1 } else { -1 }).collect();
            let q = k.exact_conditional(&x);
            for ym in 0..NS {
                out[ym] += mu[xm] * q[ym];
            }
        }
    }
    out
}

fn kl(p: &[f64], q: &[f64]) -> f64 {
    p.iter()
        .zip(q)
        .map(|(&a, &b)| if a > 0.0 { a * (a.ln() - b.max(1e-300).ln()) } else { 0.0 })
        .sum()
}

fn main() {
    // patch: 4 in, 4 out, 4 hidden on a 4x3 Z1-topology patch (12 sites, all used)
    let roles = [1u8, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3, 2];

    // initial input distribution: peaked, full support
    let mut mu0 = vec![0.0f64; NS];
    for m in 0..NS {
        mu0[m] = (0.55f64).powi(m.count_ones() as i32);
    }
    let z: f64 = mu0.iter().sum();
    for v in mu0.iter_mut() {
        *v /= z;
    }

    // ---- compile each stage under the TARGET program's exact input marginal (context matching) ----
    let mut kernels = Vec::new();
    let mut eps = Vec::new();
    let mut mu_target = mu0.clone();
    println!("compiling a 3-stage program onto Z1-patch kernels (4 in / 4 out / 4 hidden each):");
    for stage in 0..3 {
        let cpt = stage_cpt(stage);
        let mut k = patch_kernel(4, 3, &roles, 1.0, 0xC0DE + stage as u64);
        let (ce0, ce1) = fit(&mut k, &cpt, &mu_target, 1500, 0.08);
        let e = factor_eps(&k, &cpt, &mu_target);
        println!("  stage {stage}: CE {ce0:.3} -> {ce1:.3}, per-factor eps_{stage} = {e:.4} nats");
        eps.push(e);
        kernels.push(k);
        mu_target = push(&mu_target, &cpt);
    }

    // ---- exact readout KL vs the bound ----
    let mut mu_t = mu0.clone();
    let mut mu_c = mu0.clone();
    for stage in 0..3 {
        mu_t = push(&mu_t, &stage_cpt(stage));
        mu_c = push_kernel(&mu_c, &kernels[stage]);
    }
    let readout = kl(&mu_t, &mu_c);
    let bound: f64 = eps.iter().sum();
    println!("\n  exact readout KL(target || compiled) = {readout:.4} nats");
    println!("  bound  sum_l eps_l                   = {bound:.4} nats");
    let bound_ok = readout <= bound + 1e-9;
    println!("  Eq.17 bound: {}", if bound_ok { "HOLDS (exactly, no sampling)" } else { "VIOLATED — bug" });
    println!("  slack = {:.1}% of the bound (mixing kernels contract errors; the bound is worst-case)",
             100.0 * (1.0 - readout / bound.max(1e-12)));

    // ---- context matching: compile stage 2 under uniform inputs instead of its true marginal ----
    let cpt2 = stage_cpt(2);
    let mut mu2_true = mu0.clone();
    for stage in 0..2 {
        mu2_true = push(&mu2_true, &stage_cpt(stage));
    }
    let uniform = vec![1.0 / NS as f64; NS];
    let mut k_ctx = patch_kernel(4, 3, &roles, 1.0, 0x77);
    let mut k_uni = patch_kernel(4, 3, &roles, 1.0, 0x77); // same init, only mu differs
    fit(&mut k_ctx, &cpt2, &mu2_true, 1500, 0.08);
    fit(&mut k_uni, &cpt2, &uniform, 1500, 0.08);
    let e_ctx = factor_eps(&k_ctx, &cpt2, &mu2_true);
    let e_uni = factor_eps(&k_uni, &cpt2, &mu2_true); // evaluated under the TRUE inputs
    println!("\ncontext matching (stage 2, same init, only the training input distribution differs):");
    println!("  eps under true inputs — compiled on true marginal: {e_ctx:.4}   compiled on uniform: {e_uni:.4}");
    let ctx_ok = e_ctx <= e_uni + 1e-6;
    println!("  {}", if ctx_ok {
        "context-matched compilation is at least as good on the inputs the program actually feeds it"
    } else {
        "UNEXPECTED: uniform beat context-matched on true inputs"
    });

    let ok = bound_ok && ctx_ok;
    println!("\n  verdict: {}", if ok { "PASS — the compilation bound and the context-matching effect are verified exactly" } else { "FAIL" });
    std::process::exit(if ok { 0 } else { 1 });
}
