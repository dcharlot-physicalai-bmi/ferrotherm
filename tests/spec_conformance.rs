//! The `.ftp` v1 test vectors, from `spec/ftp-v1.md` §8.
//!
//! This file exists to make the specification authoritative over the implementation rather than the
//! other way round. Each case is transcribed from the table in the spec; if one fails, either the
//! implementation is wrong or the spec is, and the spec says which:
//!
//! > Where this document and any implementation disagree, this document is correct.

use ferrotherm::ftp::Program;

fn rejects(src: &str, line: usize, needle: &str) {
    match Program::from_ftp(src) {
        Ok(_) => panic!("spec requires rejection of:\n{src}"),
        Err(e) => {
            assert_eq!(e.line, line, "spec requires line {line} for:\n{src}\ngot: {e}");
            assert!(
                e.message.to_lowercase().contains(&needle.to_lowercase()),
                "message should mention {needle:?}, got: {e}"
            );
        }
    }
}

#[test]
fn vector_1_ftp_must_come_first() {
    rejects("spins 4\n", 1, "ftp");
}

#[test]
fn vector_2_unsupported_version_is_rejected_and_reported() {
    rejects("ftp 99\n", 1, "99");
}

#[test]
fn vector_3_spins_must_precede_a_spin_reference() {
    rejects("ftp 1\nbias 0 1\n", 2, "must appear before");
}

#[test]
fn vector_4_an_out_of_range_index_is_rejected() {
    rejects("ftp 1\nspins 4\nfactor 1 0 9\n", 3, "out of range");
}

#[test]
fn vector_5_a_repeated_variable_is_rejected() {
    // s*s = 1, so this silently changes the factor's order.
    rejects("ftp 1\nspins 4\nfactor 1 0 0\n", 3, "appears 2 times");
}

#[test]
fn vector_6_an_unknown_directive_is_rejected() {
    rejects("ftp 1\nspins 4\nwobble 3\n", 3, "unknown directive");
}

#[test]
fn vector_7_an_unknown_encoding_is_rejected() {
    rejects("ftp 1\nspins 4\nencode 0 3 trinary\n", 3, "unknown encoding");
}

#[test]
fn vector_8_an_unknown_observe_token_is_accepted_and_ignored() {
    // Deliberately the opposite of vector 6: tolerating new observables is what lets the format
    // gain them without a version bump.
    let p = Program::from_ftp("ftp 1\nspins 4\nobserve entropy\n")
        .expect("an unknown observe token must be accepted");
    assert_eq!(p.observe, vec!["entropy".to_string()]);
}

#[test]
fn vector_9_spins_is_required() {
    rejects("ftp 1\n", 0, "spins");
}

#[test]
fn vector_10_the_documented_example_parses_and_has_ground_energy_minus_three() {
    let src = "ftp 1\nname frustrated-ring\nspins 5\n\
               factor -1 0 1\nfactor -1 1 2\nfactor -1 2 3\nfactor -1 3 4\nfactor -1 4 0\n\
               stage 0.05 40 1 1\nstage 4 40 1 1\nobserve energy\ntarget cpu\nprice z1_spice\n";
    let p = Program::from_ftp(src).expect("the documented example must parse");
    assert_eq!(p.spins, 5);
    assert_eq!(p.factors.len(), 5);
    assert_eq!(p.schedule.len(), 2);
    let g = p.to_graph().unwrap();
    let e = ferrotherm::oracle::Solver::solve(&ferrotherm::oracle::Exhaustive, &g).1;
    assert_eq!(e, -3.0, "an odd antiferromagnetic ring cannot do better than -3");
}

#[test]
fn vector_11_repeated_biases_accumulate() {
    let p = Program::from_ftp("ftp 1\nspins 4\nbias 2 0.1\nbias 2 0.1\n").unwrap();
    let g = p.to_graph().unwrap();
    // the meaning accumulates, whatever the document's line structure
    assert!((g.h[2] - 0.2).abs() < 1e-12, "biases must sum, got {}", g.h[2]);
}

#[test]
fn vector_12_awkward_floats_round_trip_bit_identically() {
    let awkward = [0.1, 1.0 / 3.0, 1e-300, 1e300, core::f64::consts::PI, -2.220446049250313e-16];
    let mut src = format!("ftp 1\nspins {}\n", awkward.len() + 1);
    for (i, w) in awkward.iter().enumerate() {
        src.push_str(&format!("factor {w} {i} {}\n", i + 1));
    }
    let p = Program::from_ftp(&src).unwrap();
    for (f, want) in p.factors.iter().zip(awkward.iter()) {
        assert_eq!(f.weight().to_bits(), want.to_bits(), "a coupling lost bits");
    }
    // and a write of a parse is byte-identical to a re-parse of that write
    let out = p.to_ftp();
    assert_eq!(Program::from_ftp(&out).unwrap().to_ftp(), out);
}

#[test]
fn vector_13_comments_and_blank_lines_do_not_change_the_digest() {
    let clean = "ftp 1\nspins 5\nfactor -1 0 1\nfactor -1 1 2\nstage 0.05 40 1 1\n";
    let noisy = "# a program\n\nftp 1\n\n  spins 5   # five spins\nfactor -1 0 1\n\n\
                 factor -1 1 2\nstage 0.05 40 1 1\n\n# end\n";
    let (a, b) = (Program::from_ftp(clean).unwrap(), Program::from_ftp(noisy).unwrap());
    assert_eq!(a.digest(), b.digest(), "comments must not change the digest");
    assert_eq!(a, b);
}

#[test]
fn the_sign_convention_is_what_the_spec_says() {
    // Section 4 is normative: a positive weight prefers the product to be +1, so a ferromagnetic
    // bond is LOW energy when aligned. Bridging to a format with the opposite convention has to
    // negate at the boundary, and getting it backwards is wrong on every problem while looking fine.
    let p = Program::from_ftp("ftp 1\nspins 2\nfactor 1 0 1\n").unwrap();
    let g = p.to_graph().unwrap();
    assert_eq!(g.energy(&[1, 1]), -1.0, "aligned must be low energy");
    assert_eq!(g.energy(&[1, -1]), 1.0);
}
