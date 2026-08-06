//! A `.ftp` written by the browser, parsed on the CPU.
//!
//! `tests/fixtures/browser-lattice32.ftp` was not written by this crate. It was emitted by the
//! JavaScript writer in `docs/ide.html`, captured from a real browser run, and committed verbatim.
//! That is the point: a format with one implementation is a data structure, not a format. This test
//! is the only thing keeping the two writers honest with each other.

use ferrotherm::ftp::Program;
use ferrotherm::rng::Pcg;

const BROWSER: &str = include_str!("fixtures/browser-lattice32.ftp");

#[test]
fn the_browser_writes_a_program_the_cpu_can_read() {
    let p = Program::from_ftp(BROWSER).expect("browser output must parse");
    assert_eq!(p.spins, 1024);
    assert_eq!(p.factors.len(), 2048, "a periodic 32x32 lattice has 2 bonds per site");
    assert_eq!(p.name.as_deref(), Some("lattice2d"));
    assert_eq!(p.target.as_deref(), Some("wasm"));
}

#[test]
fn it_is_the_same_model_the_crate_would_have_built() {
    // Not "it parses" but "it means the same thing": the energy function must agree everywhere.
    let theirs = Program::from_ftp(BROWSER).unwrap().to_graph().unwrap();
    let ours = ferrotherm::ising::lattice2d(32, 1.0);
    assert_eq!(theirs.n, ours.n);

    let mut rng = Pcg::new(11, 0);
    for _ in 0..200 {
        let s: Vec<i8> = (0..ours.n).map(|_| if rng.f64() < 0.5 { 1 } else { -1 }).collect();
        assert!(
            (theirs.energy(&s) - ours.energy(&s)).abs() < 1e-12,
            "the browser's lattice and ours disagree on an energy"
        );
    }
    // and the ground state is the one a ferromagnet should have: every bond satisfied
    let all_up = vec![1i8; ours.n];
    assert_eq!(theirs.energy(&all_up), -2048.0);
}

#[test]
fn a_browser_program_round_trips_through_the_rust_writer() {
    // Read what the browser wrote, write it again from Rust, read that: the program must be
    // unchanged and its digest stable. This is what "runs unchanged, and its hash matches" means.
    let once = Program::from_ftp(BROWSER).unwrap();
    let twice = Program::from_ftp(&once.to_ftp()).unwrap();
    assert_eq!(once, twice);
    assert_eq!(once.digest(), twice.digest());
}

#[test]
fn it_samples() {
    // The end of the path: browser text in, thermodynamics out.
    let g = Program::from_ftp(BROWSER).unwrap().to_graph().unwrap();
    let sched = ferrotherm::schedule::Schedule::geometric(0.05, 1.0, 30, 20);
    let (_s, e) = ferrotherm::tempering::anneal_scheduled(&g, &sched, 3, None);
    assert!(e < -1800.0, "a 32x32 ferromagnet should anneal near its -2048 ground state, got {e}");
}
