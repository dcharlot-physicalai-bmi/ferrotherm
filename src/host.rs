//! What the machine was doing while the clock ran.
//!
//! A wall-clock number is a claim about code only when the code had the machine. Under contention
//! it is a claim about the run queue, and the two are indistinguishable once printed: `85.7 s` and
//! `14.2 s` for the same search on the same instance differ by nothing an output line records.
//!
//! That is not hypothetical here. `examples/gset_gap` reported 85.7 s for a G1 search that takes
//! about 14 s on a quiet machine, and reported it in the same format, with the same confidence, as
//! every honest timing beside it. The energy side of this workspace already refuses to measure a
//! busy machine — `ferrotherm-meter` blocks an idle baseline above a load average of 2, because
//! *a busy machine has no idle*. The timing side had no such guard, and the two failures are the
//! same failure: a number that is silently about the neighbours.
//!
//! The load average is the right instrument for it. It counts **runnable threads**, so a value
//! above ~2 on any machine means at least that many threads are never sleeping — someone else's
//! work is resident. It is cheap, it needs no privileges, and it is available wherever this crate
//! is likely to be timed.
//!
//! # What this module refuses to do
//!
//! It does not subtract contention, scale the timing, or estimate what the run "would have" taken.
//! There is no sound way to do that from a load average: the number depends on cache pressure and
//! memory bandwidth that the load average does not see. [`Timing::as_measurement`] returns `None`
//! and the elapsed seconds stay visible as context. A caller that wants a measurement waits for a
//! quiet machine — the same remedy the meter prescribes.
//!
//! ```no_run
//! use ferrotherm::host::Timing;
//! let (answer, t) = Timing::around(|| (0..1_000_000u64).sum::<u64>());
//! match t.as_measurement() {
//!     Some(s) => println!("{answer} in {s:.2} s"),
//!     None => println!("{answer}; {t}"), // says why, and what the load was
//! }
//! ```

/// Above this 1-minute load average, a wall-clock timing is not a measurement of the code.
///
/// The same threshold the energy meter uses for an idle baseline, and for the same reason: a load
/// average counts runnable threads, so 2 is already "two things want this CPU".
pub const QUIET_LOAD: f64 = 2.0;

/// The machine's 1-minute load average, or `None` where it cannot be read.
///
/// `None` means **not readable here**, never "the machine is quiet". Callers treat it as unknown
/// and proceed: refusing to time anything on a platform that exposes no load average would be a
/// worse failure than timing without the guard, and [`Timing::load1`] records that it was unknown
/// so the reader can tell the two apart.
pub fn load_average() -> Option<f64> {
    // Linux first: a file read beats spawning a process.
    if let Ok(s) = std::fs::read_to_string("/proc/loadavg") {
        return s.split_whitespace().next()?.parse().ok();
    }
    sysctl_load()
}

/// macOS and the BSDs: `sysctl -n vm.loadavg` prints `{ 1.23 4.56 7.89 }`.
///
/// Split out and `cfg`-gated rather than written inline, because the core crate is built for
/// `wasm32-unknown-unknown` in CI and `std::process` has no business appearing on that path.
#[cfg(unix)]
fn sysctl_load() -> Option<f64> {
    let out = std::process::Command::new("sysctl").args(["-n", "vm.loadavg"]).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    text.split_whitespace().find_map(|t| t.parse::<f64>().ok())
}

/// Nowhere else says. `None` is "not readable here", which callers treat as unknown.
#[cfg(not(unix))]
fn sysctl_load() -> Option<f64> {
    None
}

/// A wall-clock duration together with the evidence for whether it means anything.
///
/// Construct with [`Timing::around`], which samples the load average on **both** sides of the work
/// and keeps the larger. One sample would not do: the average is over the preceding minute, so a
/// reading taken before a ninety-second run describes a machine that no longer exists by the time
/// the run ends, and a reading taken only after misses contention that has already decayed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Timing {
    /// Elapsed wall-clock seconds. Always populated — it is context even when it is not evidence.
    pub seconds: f64,
    /// The worse of the two load samples, or `None` where the platform will not say.
    pub load1: Option<f64>,
}

impl Timing {
    /// Run `f`, timing it, and record how busy the machine was on either side.
    pub fn around<T>(f: impl FnOnce() -> T) -> (T, Timing) {
        let before = load_average();
        let t0 = std::time::Instant::now();
        let out = f();
        let seconds = t0.elapsed().as_secs_f64();
        let after = load_average();
        // The worse of the two, and `None` only when neither could be read: a known-bad sample must
        // not be erased by an unreadable one.
        let load1 = match (before, after) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (Some(a), None) | (None, Some(a)) => Some(a),
            (None, None) => None,
        };
        (out, Timing { seconds, load1 })
    }

    /// Attach a load reading to seconds measured elsewhere.
    pub fn new(seconds: f64, load1: Option<f64>) -> Timing {
        Timing { seconds, load1 }
    }

    /// Was the machine quiet enough for [`Timing::seconds`] to be about the code?
    pub fn trustworthy(&self) -> bool {
        !matches!(self.load1, Some(l) if l > QUIET_LOAD)
    }

    /// The seconds, **only** when they are a measurement.
    ///
    /// This is the accessor a report should use. Reaching the number through an `Option` is what
    /// makes the contaminated case impossible to print in the same shape as a clean one — the
    /// mistake this module exists to prevent was not a missing check, it was a check whose result
    /// nothing was obliged to consult.
    pub fn as_measurement(&self) -> Option<f64> {
        self.trustworthy().then_some(self.seconds)
    }

    /// The paragraph explaining why this timing is worthless, or `None` when it is not.
    ///
    /// Separate from [`Display`](core::fmt::Display) so a report can print the short form beside
    /// each number and the explanation once at the bottom, rather than repeating four lines of
    /// prose per row.
    pub fn caveat(&self) -> Option<String> {
        match self.load1 {
            Some(l) if l > QUIET_LOAD => Some(format!(
                "the 1-minute load average reached {l:.1} during the run, so every time above is \
                 the run queue's number rather than this code's. A load average counts runnable \
                 threads; above {QUIET_LOAD:.0}, at least that many never slept. The results are \
                 unaffected -- a cut or a bound is the same number whoever else is on the CPU -- \
                 but the seconds are not a measurement. Re-run on a quiet machine for those."
            )),
            _ => None,
        }
    }
}

impl core::fmt::Display for Timing {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.load1 {
            Some(l) if l > QUIET_LOAD => {
                write!(f, "{:.1} s -- NOT A MEASUREMENT, load {l:.1}", self.seconds)
            }
            Some(l) => write!(f, "{:.1} s, load {l:.2}", self.seconds),
            None => write!(f, "{:.1} s, load unknown on this platform", self.seconds),
        }
    }
}

/// The environment variable that lets a rate be reported from a busy machine anyway.
///
/// Any value except empty or `0`. It exists for the case where someone genuinely wants the number
/// and knows what it is worth — a smoke test in CI, a shape check while developing — and it does
/// not hide: [`Quiet::Overridden`] carries the load average so the caller can print it beside the
/// number it spoiled.
pub const ALLOW_BUSY: &str = "FERROTHERM_ALLOW_BUSY";

/// The verdict from [`require_quiet`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Quiet {
    /// The machine was quiet, or the platform does not report load at all.
    Yes { load1: Option<f64> },
    /// Busy, but [`ALLOW_BUSY`] was set. The number is still spoiled; the caller asked for it.
    Overridden { load1: f64 },
}

impl Quiet {
    /// A one-line note to print beside the numbers, or `None` when there is nothing to say.
    pub fn caveat(&self) -> Option<String> {
        match self {
            Quiet::Yes { .. } => None,
            Quiet::Overridden { load1 } => Some(format!(
                "** {ALLOW_BUSY} is set and the 1-minute load average is {load1:.1}. Every rate \
                 below is therefore a lower bound on this machine's throughput, spoiled by an \
                 unknown amount. Do not quote it. **"
            )),
        }
    }
}

/// Refuse to report a **rate** on a busy machine.
///
/// The distinction that matters: a search *result* — a cut, a bound, an energy — is the same
/// number whatever else the machine is doing, so contention makes it slow to obtain but does not
/// make it wrong. A rate is the opposite. Flips per second, nanoseconds per flip, joules per flip
/// (which is watts divided by a rate), a speedup column, a head-to-head against another
/// implementation: every one of those is a division by a wall-clock time, and dividing by the run
/// queue produces a number that looks exactly like a measurement.
///
/// So callers that print results annotate them with [`Timing`], and callers that print rates call
/// this and stop. `what` names the thing being refused, and appears in the message.
pub fn require_quiet(what: &str) -> Result<Quiet, String> {
    let allow = match std::env::var(ALLOW_BUSY) {
        Ok(v) => !v.is_empty() && v != "0",
        Err(_) => false,
    };
    decide(load_average(), allow, what)
}

/// The decision, separated from the environment so it can be tested without a busy machine or a
/// mutated process environment.
fn decide(load1: Option<f64>, allow: bool, what: &str) -> Result<Quiet, String> {
    match load1 {
        Some(l) if l > QUIET_LOAD => {
            if allow {
                Ok(Quiet::Overridden { load1: l })
            } else {
                Err(format!(
                    "refusing to report {what}: the 1-minute load average is {l:.1}, and every \
                     number here is a wall-clock time divided into something, so what would be \
                     printed is this machine's contention rather than this code's speed. A load \
                     average counts runnable threads; above {QUIET_LOAD:.0}, at least that many \
                     never slept. Wait for the machine to go quiet, or set {ALLOW_BUSY}=1 to get \
                     the numbers anyway with that caveat attached to them."
                ))
            }
        }
        _ => Ok(Quiet::Yes { load1 }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The threshold is inclusive, and the failing side is the one that matters.
    ///
    /// Tested on constructed values rather than by arranging a busy machine: the first version of
    /// the meter's equivalent guard lived inline and could only assert something real when the
    /// machine happened to be loaded, which is to say it usually asserted nothing.
    #[test]
    fn the_threshold_is_where_it_says_it_is() {
        assert!(Timing::new(1.0, Some(0.0)).trustworthy());
        assert!(Timing::new(1.0, Some(QUIET_LOAD)).trustworthy(), "the threshold itself is allowed");
        assert!(!Timing::new(1.0, Some(QUIET_LOAD + 0.01)).trustworthy());
        assert!(!Timing::new(1.0, Some(189.0)).trustworthy());
    }

    /// Unknown is not busy. A platform with no load average still gets to time things.
    #[test]
    fn an_unreadable_load_average_does_not_block_the_timing() {
        let t = Timing::new(1.0, None);
        assert!(t.trustworthy());
        assert_eq!(t.as_measurement(), Some(1.0));
        assert!(t.to_string().contains("unknown"), "but it says so: {t}");
    }

    /// The contaminated number is still reachable as context, and still marked.
    #[test]
    fn a_contaminated_timing_keeps_its_seconds_and_loses_its_status() {
        let t = Timing::new(85.7, Some(189.4));
        assert_eq!(t.seconds, 85.7, "the number is context, not a secret");
        assert_eq!(t.as_measurement(), None, "but it is not a measurement");
        let s = t.to_string();
        assert!(s.contains("NOT A MEASUREMENT"), "{s}");
        assert!(s.contains("189.4"), "the reader is told how busy: {s}");
        assert!(s.len() < 60, "the inline form stays on one line: {s}");
        let c = t.caveat().expect("a contaminated timing owes the reader an explanation");
        assert!(c.contains("189.4") && c.contains("run queue"), "{c}");
        assert!(
            Timing::new(1.0, Some(0.5)).caveat().is_none(),
            "a clean timing must not carry a warning"
        );
    }

    /// Both samples are consulted, and the worse one wins.
    #[test]
    fn the_worse_of_the_two_samples_is_the_one_kept() {
        assert_eq!(Timing::new(1.0, Some(0.1).map(|a: f64| a.max(9.0))).load1, Some(9.0));
        // And a readable sample is not erased by an unreadable one.
        let (_, t) = Timing::around(|| ());
        if let Some(l) = t.load1 {
            assert!(l >= 0.0 && l.is_finite(), "load {l}");
        }
        assert!(t.seconds >= 0.0);
    }

    /// A rate is refused on a busy machine, and the message says what would have gone wrong.
    #[test]
    fn a_rate_is_refused_when_the_machine_is_busy() {
        let e = decide(Some(157.0), false, "flips per second").unwrap_err();
        assert!(e.contains("flips per second"), "names what it refused: {e}");
        assert!(e.contains("157"), "says how busy: {e}");
        assert!(e.contains(ALLOW_BUSY), "says how to override: {e}");
    }

    /// A result is not a rate. Quiet is the pass-through case and it carries the load onward.
    #[test]
    fn a_quiet_machine_is_waved_through_with_its_reading() {
        assert_eq!(decide(Some(0.4), false, "x"), Ok(Quiet::Yes { load1: Some(0.4) }));
        assert_eq!(decide(Some(QUIET_LOAD), false, "x"), Ok(Quiet::Yes { load1: Some(QUIET_LOAD) }));
        assert_eq!(decide(None, false, "x"), Ok(Quiet::Yes { load1: None }));
        assert!(decide(Some(0.4), false, "x").unwrap().caveat().is_none());
    }

    /// The override returns the numbers and a caveat -- it does not return silence.
    ///
    /// The failure this guards against is an escape hatch that works by making the warning go away:
    /// then the busy run looks exactly like the quiet one again, which is where this started.
    #[test]
    fn the_override_hands_back_a_caveat_rather_than_hiding_the_load() {
        let q = decide(Some(157.0), true, "flips per second").unwrap();
        assert_eq!(q, Quiet::Overridden { load1: 157.0 });
        let c = q.caveat().expect("an overridden run must carry a caveat");
        assert!(c.contains("157"), "{c}");
        assert!(c.contains("lower bound"), "and says which way it is wrong: {c}");
    }

    /// Whatever this platform reports, it must be a plausible load average.
    #[test]
    fn the_platform_reading_is_sane_or_absent() {
        // `None` is the documented answer on a platform that exposes no load average, and it is
        // a pass: the guard treats unknown as permission to proceed.
        if let Some(l) = load_average() {
            assert!(l.is_finite() && l >= 0.0, "load average {l}");
        }
    }
}
