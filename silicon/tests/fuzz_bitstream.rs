//! Hostile input to the bitstream parsers: an `Err` or a shrug, never a panic.
//!
//! `silicon/` decodes a vendor binary format (Xilinx UG470) by hand, which is structurally the same
//! situation that produced six defects in the OMMX bridge: an implementation of someone else's wire
//! format, grown against the examples that turned up. It also carries the same fingerprints —
//! direct indexing, `as usize` on counts read from input, `Option` returns.
//!
//! Reading the code suggested the guards were sound. Reading the code is what missed the OMMX
//! allocation defects too, so this asks empirically instead.
//!
//! Capped allocator for the reason recorded in `tests/fuzz_parsers.rs`: an unbounded allocation is
//! only "found" by a machine actually trying to serve it.

use std::alloc::{GlobalAlloc, Layout, System};
use std::panic::{catch_unwind, AssertUnwindSafe};

const ALLOC_CAP: usize = 512 * 1024 * 1024;
struct Capped;
unsafe impl GlobalAlloc for Capped {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if l.size() > ALLOC_CAP { return core::ptr::null_mut(); }
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) { unsafe { System.dealloc(p, l) } }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        if n > ALLOC_CAP { return core::ptr::null_mut(); }
        unsafe { System.realloc(p, l, n) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        if l.size() > ALLOC_CAP { return core::ptr::null_mut(); }
        unsafe { System.alloc_zeroed(l) }
    }
}
#[global_allocator]
static ALLOC: Capped = Capped;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0
    }
    fn below(&mut self, n: usize) -> usize { if n == 0 { 0 } else { (self.next() % n as u64) as usize } }
    fn bytes(&mut self, n: usize) -> Vec<u8> { (0..n).map(|_| (self.next() & 0xFF) as u8).collect() }
}

fn rounds(d: u64) -> u64 {
    std::env::var("FT_FUZZ_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn no_panic<T>(what: &str, seed: u64, input: &[u8], f: impl FnOnce() -> T) {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let r = catch_unwind(AssertUnwindSafe(f));
    std::panic::set_hook(prev);
    if r.is_err() {
        let hex: Vec<String> = input.iter().take(48).map(|b| format!("{b:02x}")).collect();
        panic!("{what} PANICKED on {} bytes (seed {seed})\n  {}", input.len(), hex.join(" "));
    }
}

/// A minimal well-formed `.bit` container, to mutate.
fn seed_bit() -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(&[0x00, 0x09]);                       // field 0 length
    v.extend_from_slice(&[0x0f, 0xf0, 0x0f, 0xf0, 0x0f, 0xf0, 0x0f, 0xf0, 0x00]);
    v.extend_from_slice(&[0x00, 0x01]);                       // marker
    for (tag, s) in [(b'a', "design"), (b'b', "part"), (b'c', "2026/08/17"), (b'd', "14:00:00")] {
        v.push(tag);
        v.extend_from_slice(&(s.len() as u16 + 1).to_be_bytes());
        v.extend_from_slice(s.as_bytes());
        v.push(0);
    }
    v.push(b'e');
    v.extend_from_slice(&8u32.to_be_bytes());
    v.extend_from_slice(&[0xAA, 0x99, 0x55, 0x66, 0x20, 0x00, 0x00, 0x00]);
    v
}

#[test]
fn the_bit_container_parser_never_panics() {
    let mut rng = Rng(0xA13C_5F27_9D04_E6B8);
    let seed = seed_bit();

    // The seed itself must parse, or the mutations below are noise around a broken baseline.
    let f = ferrotherm_silicon::bitstream::parse_bit(&seed);
    assert_eq!(f.design, "design", "the baseline .bit must parse: {f:?}");
    assert_eq!(f.config.len(), 8, "and yield its config payload");

    for round in 0..rounds(4000) {
        // Pure noise.
        let n = rng.below(128);
        let raw = rng.bytes(n);
        no_panic("parse_bit(noise)", round, &raw, || {
            let b = ferrotherm_silicon::bitstream::parse_bit(&raw);
            let _ = ferrotherm_silicon::bitstream::find_sync(b.config);
            let _ = b.design.len();
        });

        // A real container with one thing broken -- the shape that reaches past the header.
        let mut v = seed.clone();
        match rng.below(4) {
            0 => { let i = rng.below(v.len()); v[i] = (rng.next() & 0xFF) as u8; }
            1 => { let n = rng.below(v.len()); v.truncate(n); }
            2 => { let i = rng.below(v.len()); for k in 0..rng.below(6) + 1 {
                       if i + k < v.len() { v[i + k] = 0xFF; } } }
            _ => { let i = rng.below(v.len()); v.splice(i..i, [0xFF, 0xFF, 0xFF, 0xFF]); }
        }
        no_panic("parse_bit(mutated)", round, &v, || {
            let b = ferrotherm_silicon::bitstream::parse_bit(&v);
            let _ = ferrotherm_silicon::bitstream::find_sync(b.config);
            let _ = ferrotherm_silicon::bitstream::to_words(b.config);
        });
    }
}

#[test]
fn the_packet_decoder_never_panics() {
    let mut rng = Rng(0x77E1_0B4A_3C92_D8F5);
    for round in 0..rounds(4000) {
        let n = rng.below(64);
        let words: Vec<u32> = (0..n).map(|_| (rng.next() & 0xFFFF_FFFF) as u32).collect();
        let raw: Vec<u8> = words.iter().flat_map(|w| w.to_be_bytes()).collect();
        no_panic("decode(noise)", round, &raw, || {
            let _ = ferrotherm_silicon::bitstream::decode(&words);
        });
    }

    // The counts that break arithmetic: a type-1 write and a type-2 continue both claiming far more
    // words than follow. `(i + count).min(len)` is what makes these safe, and it is worth a test
    // rather than a reading.
    for (label, w) in [
        ("type-1 write, max count", (1u32 << 29) | (2 << 27) | 0x7FF),
        ("type-2 continue, max count", (2u32 << 29) | 0x07FF_FFFF),
    ] {
        let words = vec![w];
        no_panic(label, 0, &w.to_be_bytes(), || {
            let p = ferrotherm_silicon::bitstream::decode(&words);
            assert!(!p.is_empty(), "{label} should still yield a packet");
        });
    }
}
