//! Intel RAPL: energy counters on x86, read from `/sys/class/powercap`.
//!
//! The other two backends sample POWER and multiply by time. RAPL exposes ENERGY directly, as a
//! monotonically increasing microjoule counter, so a window's cost is one subtraction rather than
//! an integral estimated from samples. That removes the failure mode both other backends have to
//! defend against — a workload shorter than the sampling interval.
//!
//! # The domain that lies
//!
//! `powercap` exposes several domains. `psys` is documented as whole-platform and is the obvious
//! one to want: it should cover the package, the memory, and the rest of the board, which is what
//! an "idle draw" argument actually needs. On the machine this was written against it is **dead**,
//! and it fails in the most dangerous possible way — readable, monotonic, plausible units, and
//! completely disconnected from the machine:
//!
//! ```text
//!            IDLE          20 CORES BUSY
//!   psys      0.207 W        0.200 W      <- does not move
//!   package   3.324 W       75.973 W
//! ```
//!
//! A meter built on the obvious choice would report that computation is free. So this checks the
//! one invariant that cannot be argued with — **the platform cannot draw less than the chip inside
//! it** — and refuses `psys` when it reads below `package`. When that happens it falls back to
//! `package-0` and says so, because package-only is a different quantity and a caller comparing it
//! against a whole-device figure needs to know which one they have.
//!
//! Package-only **understates** the machine, since it omits RAM, storage, fans and supply losses.
//! For [`duty`](ferrotherm::duty) that is the safe direction: it shrinks the incumbent's idle draw,
//! which is the term that makes the low-duty-cycle argument work, so a conclusion drawn from it is
//! conservative rather than flattering.
//!
//! # Reading it
//!
//! `energy_uj` is world-readable (`-r--r--r--`), so no privilege is needed. It wraps at
//! `max_energy_range_uj` — about 262 kJ here, roughly an hour at 75 W — and a wrap looks exactly
//! like a huge negative delta, so it is handled rather than assumed not to happen.

use std::path::{Path, PathBuf};

/// A powercap domain that can be read.
pub(crate) struct Domain {
    pub path: PathBuf,
    pub name: String,
    pub max: u64,
}

impl Domain {
    fn read(&self) -> Option<u64> {
        std::fs::read_to_string(self.path.join("energy_uj")).ok()?.trim().parse().ok()
    }

    /// Joules between two counter readings, wraparound included.
    ///
    /// `prev > now` is a wrap, not a machine that generated energy. Untreated it produces a huge
    /// negative delta once an hour, which would land as a single absurd sample in the middle of an
    /// otherwise sane run.
    pub(crate) fn delta_j(&self, prev: u64, now: u64) -> f64 {
        let d = if now >= prev { now - prev } else { (self.max - prev).saturating_add(now) };
        d as f64 / 1e6
    }
}

fn domains() -> Vec<Domain> {
    let root = Path::new("/sys/class/powercap");
    let Ok(rd) = std::fs::read_dir(root) else { return Vec::new() };
    let mut out = Vec::new();
    for e in rd.flatten() {
        let p = e.path();
        let Ok(name) = std::fs::read_to_string(p.join("name")) else { continue };
        // Only top-level domains. `intel-rapl:0:0` is the `core` SUBdomain of the package and
        // double-counts what its parent already reports.
        let base = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if base.matches(':').count() != 1 {
            continue;
        }
        let Ok(max) = std::fs::read_to_string(p.join("max_energy_range_uj")) else { continue };
        let Ok(max) = max.trim().parse::<u64>() else { continue };
        // Readability is the gate, not existence: `intel-rapl-mmio:0` is present here and its
        // counter cannot be opened.
        if std::fs::read_to_string(p.join("energy_uj")).is_err() {
            continue;
        }
        out.push(Domain { path: p, name: name.trim().to_string(), max });
    }
    out
}

/// Average watts over `secs`, for one domain.
fn watts_over(d: &Domain, secs: f64) -> Option<f64> {
    let a = d.read()?;
    let t = std::time::Instant::now();
    std::thread::sleep(std::time::Duration::from_secs_f64(secs));
    let b = d.read()?;
    Some(d.delta_j(a, b) / t.elapsed().as_secs_f64())
}

/// Pick the domain to meter, or `None` where RAPL is absent.
///
/// Prefers `psys` and **verifies it** rather than trusting it. Returns the domain and whether the
/// caller is getting whole-platform or package-only, because those are different quantities.
pub(crate) fn choose() -> Option<(Domain, &'static str)> {
    let all = domains();
    let pkg = all.iter().position(|d| d.name.starts_with("package"))?;
    let psys = all.iter().position(|d| d.name == "psys");

    if let Some(i) = psys {
        // THE INVARIANT: a platform cannot draw less than the chip inside it. Sampled briefly on
        // purpose -- this runs at `detect` time and a long probe would delay every caller.
        let (p, k) = (watts_over(&all[i], 0.4), watts_over(&all[pkg], 0.4));
        if let (Some(p), Some(k)) = (p, k) {
            if p >= k {
                let mut all = all;
                return Some((all.swap_remove(i), "psys"));
            }
        }
    }
    let mut all = all;
    Some((all.swap_remove(pkg), "package"))
}

/// What the chosen domain covers, for the `Meter`'s machine string.
pub(crate) fn scope_note(kind: &str) -> &'static str {
    match kind {
        "psys" => "RAPL psys (whole platform)",
        _ => "RAPL package only (no RAM, storage, fans or supply losses; understates the machine)",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wrap_is_not_a_machine_generating_energy() {
        // The counter wraps about once an hour at load. Untreated, the wrap lands as one absurd
        // negative sample in the middle of an otherwise sane run.
        let d = Domain { path: PathBuf::from("/nonexistent"), name: "package-0".into(), max: 1_000 };
        assert!((d.delta_j(100, 400) - 300e-6).abs() < 1e-12, "the ordinary case");
        // 900 -> 100 is a wrap: 100 uJ to the top, then 100 more.
        assert!((d.delta_j(900, 100) - 200e-6).abs() < 1e-12, "wrapped: {}", d.delta_j(900, 100));
        assert!(d.delta_j(900, 100) > 0.0, "a wrap must never produce a negative energy");
    }

    #[test]
    fn the_scope_is_named_rather_than_left_to_the_reader() {
        // Package-only and whole-platform are different quantities, and a caller comparing either
        // against a whole-DEVICE figure has to know which one they were handed.
        assert!(scope_note("psys").contains("whole platform"));
        let pkg = scope_note("package");
        assert!(pkg.contains("package only") && pkg.contains("understates"));
    }

    #[test]
    fn subdomains_are_not_offered_as_domains() {
        // `intel-rapl:0:0` is the `core` subdomain of `intel-rapl:0` and double-counts its parent.
        // Colon-count is the discriminator, so this checks the rule the enumerator applies.
        for (base, top_level) in
            [("intel-rapl:0", true), ("intel-rapl:0:0", false), ("intel-rapl:1", true), ("intel-rapl-mmio:0", true)]
        {
            assert_eq!(base.matches(':').count() == 1, top_level, "{base}");
        }
    }

    #[test]
    fn this_machine_either_has_rapl_or_says_it_does_not() {
        // Not an assertion about the platform -- an assertion that the answer is a decision rather
        // than a crash. On macOS this returns None; on the Linux box it returns a named domain.
        match choose() {
            Some((d, kind)) => {
                assert!(!d.name.is_empty());
                assert!(matches!(kind, "psys" | "package"));
                // And if psys was chosen it beat the package, which is the whole guard.
                eprintln!("RAPL domain {} ({})", d.name, scope_note(kind));
            }
            None => eprintln!("no RAPL on this platform; skipping"),
        }
    }
}
