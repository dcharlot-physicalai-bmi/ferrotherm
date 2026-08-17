//! Throw hostile input at every parser and require an `Err`, never a panic.
//!
//! # Why this exists
//!
//! An audit found **seven** ways a caller could abort the host process, and every one of them was
//! found by a person reading code. That is not a repeatable method. Each took a *small* input:
//! eleven bytes of OMMX whose length prefix wrapped `usize`, forty-five bytes of `.ftp` with a
//! colour index of `u64::MAX`, a JSON body of nested brackets, an integer range spanning most of
//! `i64`. None of them is exotic; all of them are what a fuzzer produces in its first minute.
//!
//! This matters more here than in most crates because of where the parsers sit. `ft_ommx_read` and
//! the rest are `extern "C"`, and **a Rust panic across a C ABI is non-unwinding: it aborts**. So a
//! panic in `ommx::import` is not a Rust error a caller can catch — it is the end of the process
//! that linked this library, whether that process is Python, Julia, a browser tab or a C program.
//! For these four functions, "returns `Err` on bad input" is a crash-safety property, not a matter
//! of taste.
//!
//! # What it does, and what it deliberately does not
//!
//! No fuzzing crate: the core has zero dependencies and this keeps that true. The generator is a
//! seeded xorshift, so a failure is **reproducible from the printed seed** rather than a story
//! about a run nobody can repeat.
//!
//! Three input shapes, because they find different things:
//!
//! 1. **Pure random bytes** — hits the length-prefix and bounds arithmetic.
//! 2. **Mutated valid inputs** — a real program with bytes flipped, lengths corrupted, fields
//!    truncated. This is the shape that finds the parser that accepts a header and then trusts
//!    everything after it.
//! 3. **Structured hostile values** — the specific integers that break arithmetic rather than
//!    parsing: `u64::MAX`, `i64::MIN`, `2^63`, huge counts, deep nesting. A random generator
//!    reaches these essentially never; they are where the real defects were.
//!
//! It asserts only the one thing that must always hold — **no panic** — and not that any particular
//! input is rejected. A parser is free to accept whatever it can make sense of.
//!
//! Round counts are sized to run in CI on every push rather than to be exhaustive: a few thousand
//! cases in seconds. That is the right trade for a gate — a fuzzer nobody runs finds nothing — and
//! the seeds are fixed, so raising the counts locally explores strictly more without losing what
//! this already covers.
//!
//! It earned its place on the first run, as a **hang** rather than a panic: `spins` was unbounded,
//! which made the colour-class bound `c < spins` vacuous, and fifty bytes asked for 96 GB. A second
//! run found a 1 GB allocation in `lp::parse`, which emits one objective term per value of an
//! integer's domain — during *parsing*, before `compile()` and so before the `DomainTooLarge`
//! refusal there could see it.
//!
//! `serve`'s JSON parser is fuzzed by `serve/tests/fuzz_json.rs`; it lives in another crate and
//! cannot be reached from here.

use std::alloc::{GlobalAlloc, Layout, System};
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Refuse any single allocation larger than [`ALLOC_CAP`].
///
/// **This is the guard that makes running a fuzzer on a development machine safe**, and it exists
/// because the first version of this file did not have it. A fuzzer's whole purpose is to find the
/// input a parser mishandles — and when the mishandling is an *unbounded allocation* rather than a
/// panic, "finding" it means the machine actually tries to serve the request. The first run of this
/// harness asked a laptop for **96 GB** from a fifty-byte input, and the laptop went to swap trying.
///
/// A returned null makes Rust call `handle_alloc_error`, which aborts immediately. That is a loud,
/// instant, harmless failure instead of a machine paging itself to death, and an abort naming the
/// size is enough to find the case: the seeds here are fixed, so re-running reproduces it exactly.
///
/// The cap is far above anything a legitimate parse of a small input needs, so it never fires on
/// correct behaviour — only on the class of defect this file is hunting.
struct Capped;

/// 512 MB. No parse of a sub-kilobyte input has any business allocating this much.
const ALLOC_CAP: usize = 512 * 1024 * 1024;

unsafe impl GlobalAlloc for Capped {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if l.size() > ALLOC_CAP {
            return core::ptr::null_mut();
        }
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    // Vec growth goes through realloc, so capping `alloc` alone would leave the main path open.
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        if new > ALLOC_CAP {
            return core::ptr::null_mut();
        }
        unsafe { System.realloc(p, l, new) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        if l.size() > ALLOC_CAP {
            return core::ptr::null_mut();
        }
        unsafe { System.alloc_zeroed(l) }
    }
}

#[global_allocator]
static ALLOC: Capped = Capped;

/// How many cases each loop runs. `FT_FUZZ_ROUNDS` raises it without editing this file.
///
/// The default is sized to run in CI on every push -- a fuzzer nobody runs finds nothing -- and the
/// seeds are fixed, so a deeper run explores strictly more without losing what the default covers.
/// `FT_FUZZ_ROUNDS=200000 cargo test --release --test fuzz_parsers` is the soak.
fn rounds(default: u64) -> u64 {
    std::env::var("FT_FUZZ_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Seeded xorshift64*. Deterministic, so a failure comes with the seed that reproduces it.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next() % n as u64) as usize
        }
    }
    fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| (self.next() & 0xFF) as u8).collect()
    }
}

/// Run `f` on `input` and report the input if it panics rather than returning.
///
/// The panic hook is silenced for the duration: a passing run would otherwise print hundreds of
/// backtraces from panics that are being handled, and a test whose successful output is a wall of
/// noise is a test people stop reading.
fn must_not_panic<T>(what: &str, seed: u64, input: &[u8], f: impl FnOnce() -> T) {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let outcome = catch_unwind(AssertUnwindSafe(f));
    std::panic::set_hook(prev);

    if outcome.is_err() {
        let head: Vec<String> = input.iter().take(64).map(|b| format!("{b:02x}")).collect();
        panic!(
            "{what} PANICKED on {} bytes (seed {seed}).\n  \
             A panic here aborts the host process through the C ABI -- it is not an error a caller \
             can handle.\n  first 64 bytes: {}\n  as text: {:?}",
            input.len(),
            head.join(" "),
            String::from_utf8_lossy(&input[..input.len().min(120)])
        );
    }
}

/// Valid inputs to mutate. A parser that rejects a header early is never reached by pure noise.
fn seeds_ftp() -> Vec<String> {
    vec![
        "ftp 1\nspins 4\nfactor -1 0 1\nfactor -1 1 2\n".into(),
        "ftp 1\nname x\nspins 3\nbias 0 0.5\ncolor 0 0\ncolor 1 1 2\n".into(),
        "ftp 1\nspins 8\nencode 0 4\nfactor 0.25 0 7\nschedule 0.1 3.0 10 5\n".into(),
    ]
}

fn seeds_lp() -> Vec<String> {
    vec![
        "Maximize\n  obj: x + y\nSubject To\n  c: x + y <= 2\nBinary\n  x y\nEnd\n".into(),
        "Minimize\n  obj: 3 a + 4 b\nSubject To\n  c1: a + b >= 1\nBounds\n  0 <= t <= 9\nGeneral\n  t\nBinary\n  a b\nEnd\n".into(),
    ]
}

/// Corrupt a valid input in one of several ways, each of which has found a real defect somewhere.
fn mutate(rng: &mut Rng, base: &[u8]) -> Vec<u8> {
    let mut v = base.to_vec();
    if v.is_empty() {
        return v;
    }
    match rng.below(6) {
        // flip a byte
        0 => {
            let i = rng.below(v.len());
            v[i] ^= 1 << rng.below(8);
        }
        // truncate: the classic "header parsed, body trusted"
        1 => {
            let n = rng.below(v.len());
            v.truncate(n);
        }
        // splice in high bytes, which is where length prefixes go wrong
        2 => {
            let i = rng.below(v.len());
            for k in 0..rng.below(8) + 1 {
                if i + k < v.len() {
                    v[i + k] = 0xFF;
                }
            }
        }
        // duplicate a slice
        3 => {
            let i = rng.below(v.len());
            let j = (i + rng.below(v.len() - i + 1)).min(v.len());
            let piece = v[i..j].to_vec();
            v.extend_from_slice(&piece);
        }
        // insert a hostile decimal number where a count might be parsed
        4 => {
            let big = ["18446744073709551615", "9223372036854775808", "-9223372036854775808", "1e400", "4000000000"]
                [rng.below(5)];
            let i = rng.below(v.len());
            v.splice(i..i, big.bytes());
        }
        // deep nesting
        _ => {
            let d = rng.below(200) + 1;
            let mut s = Vec::new();
            s.extend(std::iter::repeat_n(b'[', d));
            s.extend_from_slice(&v);
            s.extend(std::iter::repeat_n(b']', d));
            v = s;
        }
    }
    v
}

#[test]
fn no_parser_panics_on_random_bytes() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for round in 0..rounds(800) {
        let len = rng.below(96);
        let raw = rng.bytes(len);
        let text = String::from_utf8_lossy(&raw).to_string();

        must_not_panic("ommx::import", round, &raw, || ferrotherm::ommx::import(&raw));
        must_not_panic("ftp::from_ftp", round, &raw, || {
            ferrotherm::ftp::Program::from_ftp(&text)
        });
        must_not_panic("lp::parse", round, &raw, || ferrotherm::lp::parse(&text));
    }
}

#[test]
fn no_parser_panics_on_mutated_valid_input() {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    let ftp: Vec<Vec<u8>> = seeds_ftp().iter().map(|s| s.as_bytes().to_vec()).collect();
    let lp: Vec<Vec<u8>> = seeds_lp().iter().map(|s| s.as_bytes().to_vec()).collect();

    for round in 0..rounds(800) {
        let fi = rng.below(ftp.len());
        let f = mutate(&mut rng, &ftp[fi]);
        must_not_panic("ftp::from_ftp (mutated)", round, &f, || {
            ferrotherm::ftp::Program::from_ftp(&String::from_utf8_lossy(&f))
        });

        let li = rng.below(lp.len());
        let l = mutate(&mut rng, &lp[li]);
        must_not_panic("lp::parse (mutated)", round, &l, || {
            ferrotherm::lp::parse(&String::from_utf8_lossy(&l))
        });

        // OMMX is protobuf, so mutating a real encoding is the only way past the field headers.
        let o = mutate(&mut rng, &ommx_seed());
        must_not_panic("ommx::import (mutated)", round, &o, || ferrotherm::ommx::import(&o));
    }
}

/// A real `ommx.v1.Instance`: one binary variable, a quadratic objective, minimise.
fn ommx_seed() -> Vec<u8> {
    vec![
        0x12, 0x02, 0x10, 0x01, 0x1a, 0x0f, 0x1a, 0x0d, 0x08, 0x00, 0x10, 0x00, 0x19, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0xf0, 0xbf, 0x28, 0x01,
    ]
}

#[test]
fn no_parser_panics_on_the_values_that_break_arithmetic() {
    // The ones a random generator reaches essentially never, and where every real defect was.
    let hostile: Vec<String> = vec![
        // THE case, and it needs BOTH lines.
        //
        // A huge `spins` alone allocates nothing, and a huge colour index alone is caught by
        // `c < spins`. Together, the enormous spin count makes that bound vacuous and the resize
        // asks for ~96 GB. The first version of this list had each half separately and would not
        // have re-found the defect it was written for -- checked by reverting the fix and watching
        // this test still pass.
        "ftp 1\nspins 18446744073709551615\ncolor 4000000000 0\n".into(),
        "ftp 1\nspins 4294967295\ncolor 4000000000 0\n".into(),
        // colour index: `c + 1` overflowed
        "ftp 1\nspins 2\ncolor 18446744073709551615 0 1\n".into(),
        "ftp 1\nspins 2\ncolor 4000000000 0\n".into(),
        // spin counts and indices
        "ftp 1\nspins 18446744073709551615\n".into(),
        "ftp 1\nspins 2\nfactor 1 18446744073709551615 0\n".into(),
        "ftp 1\nspins 0\nfactor 1 0 0\n".into(),
        // a self-edge, which the builder refuses with a panic if it reaches it
        "ftp 1\nspins 2\nfactor 1 0 0\n".into(),
        // non-finite weights
        "ftp 1\nspins 2\nfactor NaN 0 1\n".into(),
        "ftp 1\nspins 2\nfactor inf 0 1\n".into(),
        "ftp 1\nspins 2\nbias 0 -inf\n".into(),
        // schedules
        "ftp 1\nspins 2\nschedule 0 0 18446744073709551615 1\n".into(),
        // encode
        "ftp 1\nspins 2\nencode 0 18446744073709551615\n".into(),
    ];
    for (i, s) in hostile.iter().enumerate() {
        must_not_panic("ftp::from_ftp (hostile)", i as u64, s.as_bytes(), || {
            ferrotherm::ftp::Program::from_ftp(s)
        });
    }

    let lp_hostile: Vec<String> = vec![
        // a range whose size overflows i64
        "Maximize\n  obj: t\nBounds\n  -4611686018427387904 <= t <= 4611686018427387904\nGeneral\n  t\nEnd\n".into(),
        "Maximize\n  obj: t\nBounds\n  -9223372036854775808 <= t <= 9223372036854775807\nGeneral\n  t\nEnd\n".into(),
        // non-finite coefficients
        "Maximize\n  obj: 1e400 x\nBinary\n  x\nEnd\n".into(),
        "Maximize\n  obj: nan x\nBinary\n  x\nEnd\n".into(),
        // a variable used but never declared
        "Maximize\n  obj: q\nBinary\n  x\nEnd\n".into(),
        // empty everything
        "Maximize\nEnd\n".into(),
        "End\n".into(),
        "".into(),
    ];
    for (i, s) in lp_hostile.iter().enumerate() {
        must_not_panic("lp::parse (hostile)", i as u64, s.as_bytes(), || ferrotherm::lp::parse(s));
    }

    // OMMX: hostile varints and lengths.
    let ommx_hostile: Vec<Vec<u8>> = vec![
        // a length prefix of ~2^64: `i + len` used to wrap past the bounds check
        vec![0x0A, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01],
        // a truncated field header
        vec![0x0A],
        vec![0x12],
        // a varint that never terminates
        vec![0xFF; 32],
        // a length that overruns by one
        vec![0x0A, 0x05, 0x01, 0x02, 0x03],
        vec![],
    ];
    for (i, b) in ommx_hostile.iter().enumerate() {
        must_not_panic("ommx::import (hostile)", i as u64, b, || ferrotherm::ommx::import(b));
    }
}

/// Anything that survives the parser must also survive being COMPILED.
///
/// Parsing is half the surface. `ft_model_integer` accepted a range spanning most of `i64` and
/// returned success -- the abort came later, in `ft_model_compile`. A fuzzer that stops at the
/// parser would have called that input clean.
#[test]
fn nothing_that_parses_panics_when_it_is_then_compiled() {
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let lp = seeds_lp();
    let mut compiled = 0u32;
    for round in 0..rounds(600) {
        let li = rng.below(lp.len());
        let l = mutate(&mut rng, lp[li].as_bytes());
        let text = String::from_utf8_lossy(&l).to_string();
        must_not_panic("lp::parse -> compile", round, &l, || {
            if let Ok(m) = ferrotherm::lp::parse(&text) {
                let _ = m.compile();
            }
        });
        if ferrotherm::lp::parse(&text).is_ok() {
            compiled += 1;
        }
    }
    // A floor. If mutation destroyed every input, this loop compiled nothing and proved nothing --
    // the same vacuous-pass shape the gates in scripts/ each have a guard against.
    assert!(
        compiled > 10,
        "only {compiled} of 600 mutated inputs still parsed; this exercised almost no compile paths"
    );
}

/// The guard guards.
///
/// A cap nobody has watched fire is a cap that might be wired to nothing — the same shape as every
/// other check in this repository that turned out to be inert. This asks for more than the cap in a
/// child process and requires that child to die rather than serve it.
#[test]
fn the_allocation_cap_actually_fires() {
    // Run in a child, because a refused allocation aborts the process by design.
    if std::env::var("FT_ALLOC_CAP_CHILD").is_ok() {
        let n = ALLOC_CAP * 4;
        let v: Vec<u8> = Vec::with_capacity(n);
        // Unreachable if the cap works; the print keeps the compiler from eliding the request.
        println!("allocated {} bytes, which the cap should have refused", v.capacity());
        return;
    }
    let exe = std::env::current_exe().expect("test binary path");
    let out = std::process::Command::new(exe)
        .args(["the_allocation_cap_actually_fires", "--exact", "--nocapture"])
        .env("FT_ALLOC_CAP_CHILD", "1")
        .output()
        .expect("re-run self");
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    assert!(
        !out.status.success(),
        "a {}-byte request must not succeed under a {ALLOC_CAP}-byte cap: {text}",
        ALLOC_CAP * 4
    );
    assert!(
        !text.contains("allocated"),
        "the allocation was served rather than refused: {text}"
    );
}
