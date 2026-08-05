// Gradient-estimator cross-validation — three independent routes to the same derivative.
//
// Circuit: PNot(bit, theta0) ; Err ; CtrlGauss(u ~ N(k*err, 0.3), k=theta1) ; Lin(x += 0.5 u).
// Loss: L = (x - 1)^2 + 0.4 * bit.
// The three estimators (REINFORCE, parameter-shift, central FD with common random numbers) have
// no shared machinery beyond the forward run, so agreement within noise validates each.
//
// run: cargo run --release --example grad_check

use ferrotherm::program::{Force, Gate, Program, State};

fn main() {
    let prog = Program {
        gates: vec![
            Gate::PNot { bit: 0, p_theta: 0 },
            Gate::Err { err: 1, x: 0, tgt: 1.0 },
            Gate::CtrlGauss { u: 2, err: 1, p_k: 1, sigma: 0.3 },
            Gate::Lin { x: 0, u: 2, a: 1.0, b: 0.5 },
        ],
        graphs: vec![],
        n_params: 2,
    };
    let init = State { bits: vec![1], reals: vec![0.2, 0.0, 0.0] };
    let params = [0.4, 0.9]; // theta (flip logit), k (gain)
    let loss = |s: &State| (s.reals[0] - 1.0).powi(2) + 0.4 * s.bits[0] as f64;

    let n = 400_000;
    let (g_rf, mean_l) = prog.reinforce_grad(&init, &params, &loss, n, 0x6AAD);
    let g_ps = prog.pshift_grad_pnot(0, &init, &params, &loss, 50_000, 0x51F7);
    let g_fd0 = prog.fd_grad(0, 0.02, &init, &params, &loss, 400_000, 0xFD0);
    let g_fd1 = prog.fd_grad(1, 0.02, &init, &params, &loss, 100_000, 0xFD1);

    println!("stochastic program, E[L] = {mean_l:.4}; gradients by three independent routes:\n");
    println!("  param        REINFORCE   param-shift    finite-diff");
    println!("  theta (flip) {:>10.4} {:>13.4} {:>14.4}", g_rf[0], g_ps, g_fd0);
    println!("  k     (gain) {:>10.4} {:>13} {:>14.4}", g_rf[1], "-", g_fd1);

    let ok0 = (g_rf[0] - g_fd0).abs() < 0.02 && (g_ps - g_fd0).abs() < 0.02;
    let ok1 = (g_rf[1] - g_fd1).abs() < 0.03;
    println!(
        "\n  verdict: {}",
        if ok0 && ok1 {
            "PASS — all estimators agree within noise; the program layer differentiates correctly"
        } else {
            "FAIL — estimator disagreement"
        }
    );
    // silence unused-import warning for Force when examples evolve
    let _ = Force::None;
    std::process::exit(if ok0 && ok1 { 0 } else { 1 });
}
