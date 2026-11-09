//! Measured joules, or an explicit absence.
//!
//! Instruments, in the order a host actually has them:
//!
//! - macOS: rootless `macmon` (`IOReport`), via `ferrotherm_meter`. No other macOS path.
//! - Linux CPU: Intel RAPL `energy_uj` under `/sys/class/powercap` when that file is readable.
//! - Linux CPU, when powercap is missing or unreadable: `perf` event `power/energy-pkg`.
//! - NVIDIA GPU: NVML `nvmlDeviceGetTotalEnergyConsumption`, only when no wall-power
//!   backend and no CPU counter opened. It is not added to a CPU counter.
//!
//! A busy machine is not given a fake baseline. The quiet threshold is
//! [`ferrotherm::host::QUIET_LOAD`] (2 runnable threads). The harness polls for a bounded
//! number of attempts and, if the machine is still above that line, records `load_gated`
//! with the loads it saw. It does not lower the threshold.
//!
//! `--contended` does not move that threshold. It records `load1` and other-process CPU
//! on every trial, interleaves ferrotherm then THRML, and reports energy above idle with
//! intervals. Every number from that mode is evidence grade `contended host`. A quiet
//! sample during that mode does not promote the run. The two grades are not pooled.
//!
//! Each trial repeats the sampler until the window is at least [`MIN_ENERGY_WINDOW_S`],
//! then divides joules, seconds, and effective-sample size by the repetition count.
//! Attribution also requires [`MIN_COUNTER_UPDATES`] advances of the counter. A
//! sub-millisecond window that catches zero or one RAPL tick is not a per-sample price.
//!
//! Summaries are per workload, sampler, instrument, and scope. RAPL package energy and
//! macmon `SoC` telemetry do not share a table. Package and core do not share one either.
//!
//! RAPL on hostname `framework-128gb-0000` is the primary energy evidence. That host
//! reads package `energy_uj` without root. The permission denial was `framework-32gb-0001`.
//! macmon is secondary. Its field `sys_power` is `SoC` telemetry, not wall power.
//!
//! Gross energy is the counter delta over the run. Incremental energy is gross minus
//! `idle_watts * seconds`. Both are stored. A delta inside the baseline's own wander is
//! reported and marked not attributable: the number exists, and it is not a per-sample price.
//!
//! Projected device prices are not an input to any function here. There is no parameter
//! for them, so they cannot be subtracted, added, or substituted when a counter is missing.

use ferrotherm::host::{load_average, QUIET_LOAD};
use ferrotherm_meter::Meter;
use std::time::Duration;

/// What a counter said, and what it covers.
#[derive(Clone, Debug)]
pub struct Joules {
    pub gross: Option<f64>,
    pub incremental: Option<f64>,
    pub idle_watts: Option<f64>,
    pub idle_sigma: Option<f64>,
    pub seconds: f64,
    pub attributable: bool,
    pub evidence: &'static str,
    pub instrument: String,
    pub scope: String,
    pub note: String,
    /// Always false. Present so a reader of the JSON can see the field and a test can lock it.
    pub projected_prices_applied: bool,
}

impl Joules {
    pub fn unavailable(seconds: f64, instrument: &str, note: &str) -> Joules {
        Joules {
            gross: None,
            incremental: None,
            idle_watts: None,
            idle_sigma: None,
            seconds,
            attributable: false,
            evidence: "unavailable",
            instrument: instrument.to_string(),
            scope: "none".into(),
            note: note.to_string(),
            projected_prices_applied: false,
        }
    }
}

/// How many baseline standard deviations an incremental must clear.
///
/// The band is `NOISE_SIGMA_K * sigma * seconds`. A reading inside it, and any negative
/// incremental, is not attributable. The 8e9ad0a scale run marked 99 negative increments
/// attributable because the old test used the absolute value.
pub const NOISE_SIGMA_K: f64 = 3.0;

/// How many baseline standard deviations the before/after idle may move.
///
/// A bracket whose two ends differ by more than this is not a baseline. The quiet-host
/// gate is a separate check and is not this number.
pub const DRIFT_SIGMA_K: f64 = 3.0;

/// `gross - idle * seconds`, and whether that delta is a positive clearance of the noise band.
///
/// The incremental is always returned, including when it is negative. `attributable` is
/// false for a negative value and for one inside `NOISE_SIGMA_K` standard deviations.
pub fn split_incremental(gross: f64, idle_watts: f64, idle_sigma: f64, seconds: f64) -> (f64, bool) {
    let incremental = gross - idle_watts * seconds;
    let band = NOISE_SIGMA_K * idle_sigma * seconds;
    let attributable = incremental > band && seconds > 0.0 && idle_sigma.is_finite() && incremental.is_finite();
    (incremental, attributable)
}

/// Mean and sigma of an idle bracket, and whether the two ends drifted too far apart.
///
/// Sigma is the root-mean-square of the two window sigmas. The drift test uses that sigma,
/// so a quiet pair of readings can fail a large jump and a noisy pair is judged against its
/// own wander.
pub fn combine_bracket(before_w: f64, before_s: f64, after_w: f64, after_s: f64) -> (f64, f64, f64, bool) {
    let mean = 0.5 * (before_w + after_w);
    let sigma = (0.5 * (before_s * before_s + after_s * after_s)).sqrt();
    let drift = (before_w - after_w).abs();
    let finite = before_w.is_finite() && after_w.is_finite() && sigma.is_finite();
    let drift_ok = finite && drift <= DRIFT_SIGMA_K * sigma;
    (mean, sigma, drift, drift_ok)
}

/// Ferrotherm incremental minus THRML incremental, both against the same bracket.
pub fn paired_delta(ferrotherm_inc: f64, thrml_inc: f64) -> f64 {
    ferrotherm_inc - thrml_inc
}

/// Whether a paired difference clears the noise of both windows.
///
/// The sign is kept. A negative difference means THRML drew more energy than ferrotherm
/// inside the bracket. That is not the negative-incremental failure, which is a side
/// whose own gross sat below the idle bracket.
pub fn paired_attributable(
    delta: f64,
    sigma: f64,
    seconds_a: f64,
    seconds_b: f64,
    drift_ok: bool,
    updates_a: u32,
    updates_b: u32,
) -> bool {
    let band = NOISE_SIGMA_K * sigma * seconds_a.hypot(seconds_b);
    drift_ok
        && updates_a >= MIN_COUNTER_UPDATES
        && updates_b >= MIN_COUNTER_UPDATES
        && delta.is_finite()
        && band.is_finite()
        && delta.abs() > band
}

/// Even pairs run ferrotherm then THRML. Odd pairs reverse that.
///
/// Five pairs therefore start A,B and then B,A, and each sampler runs five times.
pub fn pair_order(pair: usize) -> [&'static str; 2] {
    if pair.is_multiple_of(2) { ["ferrotherm", "thrml"] } else { ["thrml", "ferrotherm"] }
}

/// `0-3,8,10-11` into core indices. Empty input is an error, not "every core".
pub fn parse_cpu_list(text: &str) -> Result<Vec<usize>, String> {
    let text = text.trim();
    if text.is_empty() {
        return Err("--cpus was empty".into());
    }
    let mut out = Vec::new();
    for part in text.split(',') {
        let part = part.trim();
        if part.is_empty() {
            return Err(format!("empty field in cpu list {text}"));
        }
        if let Some((a, b)) = part.split_once('-') {
            let lo: usize = a.trim().parse().map_err(|_| format!("bad cpu {a}"))?;
            let hi: usize = b.trim().parse().map_err(|_| format!("bad cpu {b}"))?;
            if hi < lo {
                return Err(format!("cpu range {part} is backwards"));
            }
            for c in lo..=hi {
                if !out.contains(&c) {
                    out.push(c);
                }
            }
        } else {
            let c: usize = part.parse().map_err(|_| format!("bad cpu {part}"))?;
            if !out.contains(&c) {
                out.push(c);
            }
        }
    }
    if out.is_empty() {
        return Err(format!("cpu list {text} named no core"));
    }
    Ok(out)
}

/// Busy fraction of cores that are not in `pinned`, from two `/proc/stat` snapshots.
///
/// Each line is `cpuN user nice system idle iowait irq softirq steal ...`. Busy is
/// user+nice+system+irq+softirq+steal. A core with no elapsed ticks is omitted.
pub fn other_core_busy(before: &str, after: &str, pinned: &[usize]) -> Vec<(usize, f64)> {
    let mut prior = std::collections::BTreeMap::new();
    for line in before.lines() {
        if let Some((cpu, busy, total)) = parse_cpu_n(line) {
            prior.insert(cpu, (busy, total));
        }
    }
    let mut out = Vec::new();
    for line in after.lines() {
        let Some((cpu, busy, total)) = parse_cpu_n(line) else { continue };
        if pinned.contains(&cpu) {
            continue;
        }
        let Some((b0, t0)) = prior.get(&cpu) else { continue };
        let db = busy.saturating_sub(*b0) as f64;
        let dt = total.saturating_sub(*t0) as f64;
        if dt > 0.0 {
            out.push((cpu, db / dt));
        }
    }
    out
}

fn parse_cpu_n(line: &str) -> Option<(usize, u64, u64)> {
    let mut it = line.split_whitespace();
    let name = it.next()?;
    let cpu = name.strip_prefix("cpu")?.parse().ok()?;
    let mut f = [0u64; 8];
    for slot in &mut f {
        *slot = it.next()?.parse().ok()?;
    }
    let busy = f[0] + f[1] + f[2] + f[5] + f[6] + f[7];
    let total = busy + f[3] + f[4];
    Some((cpu, busy, total))
}

/// How many times to re-read the load average before calling the host load-gated.
///
/// Fifteen attempts, two seconds apart in the live path, is thirty seconds. That is the
/// bound. A machine that is still above [`QUIET_LOAD`] at the end is reported, not treated
/// as quiet.
pub const QUIET_ATTEMPTS: usize = 15;

/// Every measured trial repeats the sampler until the window is at least this long.
///
/// A ferrotherm chain on the pinned workloads returns in about a millisecond. RAPL on
/// `framework-128gb-0000` does not advance on that interval: one trial at revision
/// `189502d` recorded 0 J gross and a negative incremental and was still marked
/// attributable. Two seconds covers the counter's update cadence. The window's joules
/// are then divided by the repetition count.
pub const MIN_ENERGY_WINDOW_S: f64 = 2.0;

/// Attribution needs at least this many counter advances inside the window.
///
/// One tick, or none, is a collision with the update cadence. Eight is the same floor
/// the wall-power meter already uses for a mean.
pub const MIN_COUNTER_UPDATES: u32 = 8;

/// Joules and seconds of one repetition, from a window that ran `repetitions` times.
///
/// A zero repetition count is not a measurement.
pub fn per_repetition(window_joules: f64, window_seconds: f64, repetitions: u32) -> (f64, f64) {
    if repetitions == 0 {
        return (f64::NAN, f64::NAN);
    }
    let n = f64::from(repetitions);
    (window_joules / n, window_seconds / n)
}

/// How many times successive finite readings of a counter differed.
///
/// The first reading is the baseline, not an update. Equal neighbours are the counter
/// standing still. `NaN` is ignored.
pub fn count_counter_updates(readings: &[f64]) -> u32 {
    let mut updates = 0u32;
    let mut prev: Option<f64> = None;
    for reading in readings {
        if !reading.is_finite() {
            continue;
        }
        if let Some(p) = prev
            && *reading != p
        {
            updates += 1;
        }
        prev = Some(*reading);
    }
    updates
}

/// Gross joules from the first and last finite reading.
///
/// `max_j` is the counter's range when a wrap is defined. Without it, a decrease is not
/// a measurement.
pub fn window_gross(readings: &[f64], max_j: Option<f64>) -> f64 {
    let Some(first) = readings.iter().copied().find(|r| r.is_finite()) else {
        return f64::NAN;
    };
    let Some(last) = readings.iter().copied().rev().find(|r| r.is_finite()) else {
        return f64::NAN;
    };
    match max_j {
        Some(max) => counter_delta_j(first, last, max),
        None if last >= first => last - first,
        None => f64::NAN,
    }
}

/// Read `read` about every 250 µs while `body` runs.
///
/// The samples are the cadence record. `body` runs on this thread. `read` runs on another,
/// so it must not touch the sampler `body` is driving. On panic inside `body` the reader
/// is stopped before the scope joins it.
/// Counter samples taken while `body` ran, with the median spacing of the reads.
pub struct SampledWindow<R> {
    /// Readings in the counter's own unit. The energy is still the first-to-last delta.
    pub readings: Vec<f64>,
    /// Wall seconds `body` took.
    pub seconds: f64,
    /// Median spacing of successful reads, in seconds. `NaN` below two reads.
    pub cadence_s: f64,
    /// Whatever `body` returned.
    pub value: R,
}

pub fn sample_during<R>(
    mut read: impl FnMut() -> Option<f64> + Send,
    body: impl FnOnce() -> R,
) -> SampledWindow<R> {
    let stop = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|scope| {
        let flag = &stop;
        let handle = scope.spawn(move || {
            let mut samples = Vec::new();
            let mut times = Vec::new();
            while !flag.load(std::sync::atomic::Ordering::Relaxed) {
                if let Some(x) = read() {
                    times.push(std::time::Instant::now());
                    samples.push(x);
                }
                std::thread::sleep(Duration::from_micros(250));
            }
            (samples, times)
        });
        let _stop_on_drop = StopReader(&stop);
        let started = std::time::Instant::now();
        let value = body();
        let seconds = started.elapsed().as_secs_f64();
        drop(_stop_on_drop);
        let (readings, times) = handle.join().unwrap_or_default();
        let mut dts = Vec::new();
        for w in times.windows(2) {
            let dt = w[1].saturating_duration_since(w[0]).as_secs_f64();
            if dt > 0.0 {
                dts.push(dt);
            }
        }
        SampledWindow { readings, seconds, cadence_s: median_positive(&dts), value }
    })
}

fn median_positive(xs: &[f64]) -> f64 {
    let mut v: Vec<f64> = xs.iter().copied().filter(|x| x.is_finite() && *x > 0.0).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n.is_multiple_of(2) { 0.5 * (v[n / 2 - 1] + v[n / 2]) } else { v[n / 2] }
}

struct StopReader<'a>(&'a std::sync::atomic::AtomicBool);

impl Drop for StopReader<'_> {
    fn drop(&mut self) {
        self.0.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Package energy, core energy, and `SoC` telemetry are different measurements.
///
/// The labels that may be summarised are `package` (RAPL package or `power/energy-pkg`),
/// `core` (RAPL core, not added to package), `soc-telemetry` (`macmon` `sys_power`, not wall
/// power), `whole-system` (a legacy token; not `macmon`), `psys`, `board` (INA3221 or
/// `tegrastats` `VDD_IN`), and `gpu` (NVML). Two rows share a scope only when the label is
/// the same one of those.
pub fn scopes_may_share_a_table(a: &str, b: &str) -> bool {
    a == b && matches!(a, "package" | "core" | "soc-telemetry" | "whole-system" | "psys" | "board" | "gpu")
}

/// What the scope does not include. This is a statement, not a joule, and it is not folded
/// into the combined uncertainty.
pub fn scope_limit(scope: &str) -> &'static str {
    match scope {
        "soc-telemetry" => {
            "macmon sys_power is SoC telemetry (CPU, integrated GPU, ANE). It is not wall power. Display, fans, and the rest of the board are outside it. Package and core are different scopes and are not added here."
        }
        "package" => {
            "RAPL package energy. The core domain is inside the package and is reported as its own scope, not added here. DRAM and the board are outside this counter."
        }
        "core" => "RAPL core domain only. It is not added to package energy.",
        "psys" => "RAPL psys, kept only when it contains the package. It is not macmon and it is not a wall-plug measurement.",
        "board" => "One board rail (INA3221 or tegrastats VDD_IN). Other rails are not summed.",
        "gpu" => "NVML total energy, summed across devices. It is not package energy and not SoC telemetry.",
        "whole-system" => "Legacy label. On a macmon receipt this token meant sys_power, which is SoC telemetry, not wall power.",
        _ => "scope not recognised",
    }
}

/// Recorded quantum of a counter, in joules. `None` when the instrument does not state one.
///
/// RAPL `energy_uj` is an integer number of microjoules. NVML total energy is millijoules.
/// `macmon` emits a float and states no quantum.
pub fn counter_quantum_j(instrument: &str) -> Option<f64> {
    match instrument {
        "rapl" => Some(1e-6),
        "nvml" => Some(1e-3),
        _ => None,
    }
}

/// Both ends of a counter reading, each uncertain by one recorded quantum.
pub fn counter_quantization_j(quantum_j: f64) -> f64 {
    2.0 * quantum_j
}

/// Energy in one poll interval at the window's average power.
///
/// That is half an interval at each end. It is a bound beside the counter delta, not a
/// replacement for it. `NaN` when the cadence or the window is not a positive finite number.
pub fn counter_cadence_edge_j(gross_window: f64, window_s: f64, cadence_s: f64) -> f64 {
    if window_s <= 0.0 || !cadence_s.is_finite() || !gross_window.is_finite() || !cadence_s.is_sign_positive() {
        return f64::NAN;
    }
    cadence_s * (gross_window / window_s)
}

/// Student-t scatter and the systematic terms that sit beside it.
///
/// `combined` is the root-sum-square of the finite terms. A missing quantization or edge
/// term is omitted and stays `None`. It is not entered as zero.
#[derive(Clone, Debug)]
pub struct UncertaintyBudget {
    /// Half-width of the 95% Student-t interval, in the same units as the other terms.
    pub student_t_half: f64,
    /// Counter resolution at both ends. `None` when the instrument states no quantum.
    pub quantization: Option<f64>,
    /// Sampling-cadence and edge bound. `None` when the samples that would set it are absent.
    pub cadence_edge: Option<f64>,
    /// `idle_sigma * seconds`, one standard deviation of the baseline times the window.
    pub bracket: f64,
    /// Root-sum-square of the finite terms above.
    pub combined: f64,
}

/// Combine the stated terms. A non-finite term is left out of the root-sum-square.
pub fn combine_uncertainty(
    student_t_half: f64,
    quantization: Option<f64>,
    cadence_edge: Option<f64>,
    bracket: f64,
) -> UncertaintyBudget {
    let quantization = quantization.filter(|q| q.is_finite());
    let cadence_edge = cadence_edge.filter(|q| q.is_finite());
    let mut acc = 0.0;
    let mut n = 0u32;
    for term in [Some(student_t_half), quantization, cadence_edge, Some(bracket)].into_iter().flatten() {
        if term.is_finite() {
            acc += term * term;
            n += 1;
        }
    }
    UncertaintyBudget {
        student_t_half,
        quantization,
        cadence_edge,
        bracket,
        combined: if n == 0 { f64::NAN } else { acc.sqrt() },
    }
}

/// How long [`sample_quiet_gate`] watches the host before a run is allowed to start.
pub const QUIET_GATE_SECONDS: u64 = 60;

/// A 60-second preflight of load1 and the busy fraction of the cores outside the pin.
///
/// `held` is the unchanged quiet gate: every load sample finite and at or under
/// [`QUIET_LOAD`]. The busy fraction is recorded and is not a second threshold.
#[derive(Clone, Debug)]
pub struct QuietGate {
    /// Seconds between the first and last load sample.
    pub seconds: f64,
    /// One load1 per second, including both ends of the window.
    pub samples: Vec<f64>,
    /// Largest finite load1. `None` when none was readable.
    pub max_load1: Option<f64>,
    /// The quiet gate held for every sample.
    pub held: bool,
    /// Busy fraction of cores not in the pin, over the whole window. Empty when unpinned
    /// means every core is reported, because there is no pin to exclude.
    pub other_core_busy: Vec<(usize, f64)>,
    /// `/proc/stat` was readable at both ends. A missing file does not by itself fail the
    /// load gate; the receipt says the busy fraction was not readable.
    pub busy_readable: bool,
    /// The threshold that was applied. This is [`QUIET_LOAD`].
    pub threshold: f64,
}

/// Decide the preflight from samples already taken.
///
/// An empty series does not hold. A missing or non-finite load does not hold. A load above
/// [`QUIET_LOAD`] does not hold. The busy fraction is stored and does not change `held`.
pub fn quiet_gate_from(
    loads: &[Option<f64>],
    other_core_busy: &[(usize, f64)],
    busy_readable: bool,
    seconds: f64,
) -> QuietGate {
    let mut samples = Vec::with_capacity(loads.len());
    let mut held = !loads.is_empty();
    let mut max_l = f64::NEG_INFINITY;
    for load in loads {
        match load {
            Some(v) if v.is_finite() => {
                samples.push(*v);
                max_l = max_l.max(*v);
                if *v > QUIET_LOAD {
                    held = false;
                }
            }
            _ => {
                samples.push(f64::NAN);
                held = false;
            }
        }
    }
    QuietGate {
        seconds,
        samples,
        max_load1: if max_l.is_finite() { Some(max_l) } else { None },
        held,
        other_core_busy: other_core_busy.to_vec(),
        busy_readable,
        threshold: QUIET_LOAD,
    }
}

/// Sample load1 once a second for [`QUIET_GATE_SECONDS`], and the busy fraction of cores
/// outside `pinned` across that same window.
pub fn sample_quiet_gate(pinned: &[usize]) -> QuietGate {
    let before = std::fs::read_to_string("/proc/stat").unwrap_or_default();
    let t0 = std::time::Instant::now();
    let mut loads = Vec::with_capacity(QUIET_GATE_SECONDS as usize + 1);
    for i in 0..=QUIET_GATE_SECONDS {
        loads.push(load_average());
        if i < QUIET_GATE_SECONDS {
            std::thread::sleep(Duration::from_secs(1));
        }
    }
    let seconds = t0.elapsed().as_secs_f64();
    let after = std::fs::read_to_string("/proc/stat").unwrap_or_default();
    let busy_readable = before.contains("cpu") && after.contains("cpu");
    let busy = if busy_readable { other_core_busy(&before, &after, pinned) } else { Vec::new() };
    quiet_gate_from(&loads, &busy, busy_readable, seconds)
}

/// What runs after a quiet-gate decision.
///
/// `load1` is a one-minute average. The accuracy phase and the THRML runner are
/// CPU work, so they raise that average. A preflight that held, followed by that
/// work, then by another load check, refuses a host the preflight had accepted.
/// When the 60 s gate held, energy is next and accuracy follows it. Otherwise
/// accuracy stays first. The threshold is still [`QUIET_LOAD`]. No process is
/// removed from the load.
pub fn phase_order(quiet_gate_held: bool) -> [&'static str; 2] {
    if quiet_gate_held { ["energy", "accuracy"] } else { ["accuracy", "energy"] }
}

/// One poll of the 1-minute load average against the unchanged quiet threshold.
#[derive(Clone, Debug)]
pub struct QuietPoll {
    pub outcome: &'static str,
    pub samples: Vec<f64>,
    pub accepted: Option<f64>,
    pub threshold: f64,
}

/// `reads` are successive load averages. `NaN` is "not readable", which is not quiet.
/// The first sample at or under [`QUIET_LOAD`] accepts. If none does, the outcome is
/// `load_gated` and `accepted` is `None`.
pub fn poll_quiet(reads: &[Option<f64>]) -> QuietPoll {
    let mut samples = Vec::new();
    for read in reads {
        match read {
            Some(l) if l.is_finite() && *l <= QUIET_LOAD => {
                samples.push(*l);
                return QuietPoll { outcome: "quiet", samples, accepted: Some(*l), threshold: QUIET_LOAD };
            }
            Some(l) => samples.push(*l),
            None => samples.push(f64::NAN),
        }
    }
    QuietPoll { outcome: "load_gated", samples, accepted: None, threshold: QUIET_LOAD }
}

/// Live poll. Sleeps two seconds between attempts, at most [`QUIET_ATTEMPTS`] times.
pub fn wait_for_quiet_host() -> QuietPoll {
    let mut reads = Vec::with_capacity(QUIET_ATTEMPTS);
    for i in 0..QUIET_ATTEMPTS {
        reads.push(load_average());
        if poll_quiet(&reads).outcome == "quiet" {
            break;
        }
        if i + 1 < QUIET_ATTEMPTS {
            std::thread::sleep(Duration::from_secs(2));
        }
    }
    poll_quiet(&reads)
}

/// Mean and sample standard deviation of idle power samples. `None` below two finite samples.
pub fn idle_stats(watts: &[f64]) -> Option<(f64, f64)> {
    let v: Vec<f64> = watts.iter().copied().filter(|w| w.is_finite() && *w >= 0.0).collect();
    if v.len() < 2 {
        return None;
    }
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|w| (w - mean).powi(2)).sum::<f64>() / (n - 1.0);
    Some((mean, var.sqrt()))
}

/// Counter wrap: `max_j` is the counter's range in joules.
pub fn counter_delta_j(prev: f64, now: f64, max_j: f64) -> f64 {
    if now >= prev { now - prev } else { (max_j - prev) + now }
}

/// Watts from successive counter windows `(delta_joules, seconds)`.
pub fn watts_from_windows(windows: &[(f64, f64)]) -> Option<(f64, f64)> {
    let watts: Vec<f64> = windows.iter().filter(|(j, s)| *s > 0.0 && j.is_finite()).map(|(j, s)| j / s).collect();
    idle_stats(&watts)
}

/// Which RAPL domain is honest. `psys` is kept only when it draws at least as much as the package
/// inside it. Otherwise the package counter is the one that moves.
pub fn choose_rapl_kind(psys_w: Option<f64>, pkg_w: Option<f64>) -> Option<&'static str> {
    match (psys_w, pkg_w) {
        (Some(p), Some(k)) if p.is_finite() && k.is_finite() && p >= k => Some("psys"),
        (_, Some(k)) if k.is_finite() => Some("package"),
        _ => None,
    }
}

/// `event=0x02` or `event=0x01,umask=0x10` as a perf `config` word.
pub fn parse_perf_event(text: &str) -> Result<u64, String> {
    let mut config = 0u64;
    let mut saw = false;
    for part in text.split([',', ' ', '\n']) {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some(v) = part.strip_prefix("event=") {
            config |= parse_u64_token(v)?;
            saw = true;
        } else if let Some(v) = part.strip_prefix("umask=") {
            config |= parse_u64_token(v)? << 8;
            saw = true;
        } else if !part.contains('=') {
            config |= parse_u64_token(part)?;
            saw = true;
        }
    }
    if saw { Ok(config) } else { Err(format!("no perf event field in {text:?}")) }
}

fn parse_u64_token(v: &str) -> Result<u64, String> {
    let v = v.trim();
    if let Some(hex) = v.strip_prefix("0x").or_else(|| v.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).map_err(|_| format!("bad hex {v}"))
    } else {
        v.parse::<u64>().map_err(|_| format!("bad integer {v}"))
    }
}

/// One line of `perf stat -x ';'` output. The joule value is the field before a `Joules` unit
/// on a line that names `energy-pkg`.
pub fn parse_perf_joules(text: &str) -> Option<f64> {
    for line in text.lines() {
        if !line.contains("energy-pkg") {
            continue;
        }
        let fields: Vec<&str> = line.split([';', ',']).map(str::trim).collect();
        for (i, field) in fields.iter().enumerate() {
            if *field == "Joules"
                && let Some(prev) = i.checked_sub(1)
                && let Ok(v) = fields[prev].parse::<f64>()
                && v.is_finite()
            {
                return Some(v);
            }
        }
    }
    None
}

/// A load-gated result. Gross and incremental stay null. This is the report, not a refusal
/// that leaves the field missing.
pub fn load_gated(note: &str) -> Joules {
    Joules {
        gross: None,
        incremental: None,
        idle_watts: None,
        idle_sigma: None,
        seconds: 0.0,
        attributable: false,
        evidence: "load_gated",
        instrument: "none".into(),
        scope: "none".into(),
        note: note.to_string(),
        projected_prices_applied: false,
    }
}

/// Evidence grades a reader is allowed to see. `quiet` is only the quiet-host gate.
/// `contended host` is a different grade and is never a synonym for quiet.
pub const GRADE_QUIET: &str = "quiet";
pub const GRADE_CONTENDED: &str = "contended host";

/// `--contended` forces [`GRADE_CONTENDED`] even when the load happens to be under the gate,
/// so a contended run cannot be promoted to quiet by a lucky sample.
/// Without that flag, quiet is [`QUIET_LOAD`] and nothing above it.
pub fn grade_for(contended_mode: bool, quiet: bool) -> &'static str {
    if contended_mode {
        GRADE_CONTENDED
    } else if quiet {
        GRADE_QUIET
    } else {
        "load_gated"
    }
}

/// Quiet rows and contended rows do not share a table.
pub fn grades_may_share_a_table(a: &str, b: &str) -> bool {
    a == b && (a == GRADE_QUIET || a == GRADE_CONTENDED)
}

/// RAPL on `framework-128gb-0000` is the primary energy evidence.
/// macmon is secondary. A different Framework hostname is not the primary machine.
pub fn energy_rank(instrument: &str, hostname: &str) -> &'static str {
    let host = hostname.trim();
    if instrument == "rapl" && host == "framework-128gb-0000" {
        "primary"
    } else if instrument == "macmon" {
        "secondary"
    } else {
        "unranked"
    }
}

/// Other-process CPU time, in seconds, between two snapshots.
///
/// `busy` is every non-idle tick on the machine. `mine` is this process. The difference is
/// everyone else. A wrap or a clock that went backwards contributes zero, not a negative.
pub fn other_cpu_seconds(busy_before: u64, mine_before: u64, busy_after: u64, mine_after: u64, hz: f64) -> f64 {
    if !(hz > 0.0) {
        return f64::NAN;
    }
    let busy = busy_after.saturating_sub(busy_before) as f64;
    let mine = mine_after.saturating_sub(mine_before) as f64;
    (busy - mine).max(0.0) / hz
}

/// `cpu  user nice system idle iowait irq softirq steal ...` → busy ticks.
pub fn parse_proc_stat_busy(line: &str) -> Option<u64> {
    let mut it = line.split_whitespace();
    if it.next() != Some("cpu") {
        return None;
    }
    let mut f = [0u64; 8];
    for slot in &mut f {
        *slot = it.next()?.parse().ok()?;
    }
    Some(f[0] + f[1] + f[2] + f[5] + f[6] + f[7])
}

/// `/proc/self/stat`: utime + stime, after the command name in parentheses.
pub fn parse_self_stat_busy(text: &str) -> Option<u64> {
    let rest = text.rsplit_once(')')?.1;
    let mut it = rest.split_whitespace();
    // state ppid pgrp session tty tpgid flags minflt cminflt majflt cmajflt utime stime
    let utime: u64 = it.nth(11)?.parse().ok()?;
    let stime: u64 = it.next()?.parse().ok()?;
    Some(utime + stime)
}

/// Build the joule record for one repetition from a window and an idle baseline.
///
/// `gross` and `seconds` are already per repetition. `counter_updates` is how many times
/// the counter advanced during the whole window, before the division.
///
/// `grade` is stored as the evidence grade. A caller who passes `quiet` is asserting the
/// quiet-host gate already passed. Contended mode passes `contended host` and nothing else.
///
/// Attribution requires the incremental to clear the idle band, a strictly positive gross,
/// and at least [`MIN_COUNTER_UPDATES`] advances. Zero gross with a negative incremental
/// is the signature of a window that missed the counter, and it is not attributable.
pub fn trial_joules(
    gross: f64,
    idle_watts: f64,
    idle_sigma: f64,
    seconds: f64,
    instrument: &str,
    scope: &str,
    grade: &'static str,
    counter_updates: u32,
    drift_ok: bool,
) -> Joules {
    let (incremental, clears_band) = split_incremental(gross, idle_watts, idle_sigma, seconds);
    let cadence = counter_updates >= MIN_COUNTER_UPDATES;
    let moved = gross.is_finite() && gross > 0.0;
    let attributable = clears_band && cadence && moved && drift_ok;
    let note = if !drift_ok {
        "attribution refused: the idle bracket drifted, or the quiet gate failed on a bracket".to_string()
    } else if !cadence {
        format!(
            "attribution refused: the counter advanced {counter_updates} times and the floor is {MIN_COUNTER_UPDATES}"
        )
    } else if !moved {
        "attribution refused: gross energy was not strictly positive".to_string()
    } else if incremental.is_finite() && incremental <= 0.0 {
        "attribution refused: incremental energy is negative".to_string()
    } else if !clears_band {
        "attribution refused: incremental energy is inside the idle noise band".to_string()
    } else {
        String::new()
    };
    Joules {
        gross: Some(gross),
        incremental: Some(incremental),
        idle_watts: Some(idle_watts),
        idle_sigma: Some(idle_sigma),
        seconds,
        attributable,
        evidence: grade,
        instrument: instrument.to_string(),
        scope: scope.to_string(),
        note,
        projected_prices_applied: false,
    }
}

/// One metered trial of one sampler on one workload.
#[derive(Clone, Debug)]
pub struct EnergyTrial {
    pub workload: &'static str,
    pub side: &'static str,
    pub trial: usize,
    /// Sampler executions inside the window. Joules and `ess` are per execution.
    pub repetitions: u32,
    /// Counter advances observed in the window, before dividing by [`Self::repetitions`].
    pub counter_updates: u32,
    /// Wall time of the whole window, before dividing.
    pub window_seconds: f64,
    pub joules: Joules,
    pub ess: f64,
    pub joules_per_ess: Option<f64>,
    pub load1: Option<f64>,
    pub other_cpu_s: Option<f64>,
    pub grade: &'static str,
    pub rank: &'static str,
    /// Watts of the idle window before this pair.
    pub idle_before_w: Option<f64>,
    /// Watts of the idle window after this pair.
    pub idle_after_w: Option<f64>,
    /// Absolute drift of the bracket, in watts.
    pub baseline_drift_w: Option<f64>,
    /// Ferrotherm incremental minus THRML incremental on this scope. `None` when one side is missing.
    pub paired_delta_j: Option<f64>,
    /// The paired difference cleared drift, cadence, and noise. Sign is allowed.
    pub paired_attributable: bool,
    /// Busy fraction of cores outside the pin, `cpu:fraction` comma-separated. Empty when unpinned.
    pub other_core_busy: String,
    /// Median sample spacing over the window, in seconds. `NaN` when it was not recorded.
    pub cadence_s: f64,
    /// How the left edge of a power integral was placed. `counter` for a counter delta.
    pub left_edge: String,
    /// How the right edge was placed.
    pub right_edge: String,
    /// Seconds at the edges that the integral did not cover. Zero for a counter delta.
    pub uncovered_s: f64,
    /// Raw samples inside the window. For a counter this is the number of reads, which is not
    /// the number of times the counter changed.
    pub samples: u32,
    /// Window joules of counter quantization, both ends. `None` when the instrument states none.
    pub quantization_window_j: Option<f64>,
    /// Window joules of the cadence and edge bound. `NaN` when the samples do not state it.
    pub cadence_edge_window_j: f64,
}

impl EnergyTrial {
    pub fn from_parts(
        workload: &'static str,
        side: &'static str,
        trial: usize,
        repetitions: u32,
        counter_updates: u32,
        window_seconds: f64,
        joules: Joules,
        ess: f64,
        load1: Option<f64>,
        other_cpu_s: Option<f64>,
        grade: &'static str,
        rank: &'static str,
    ) -> EnergyTrial {
        let joules_per_ess = joules_per_ess(&joules, ess);
        EnergyTrial {
            workload,
            side,
            trial,
            repetitions,
            counter_updates,
            window_seconds,
            joules,
            ess,
            joules_per_ess,
            load1,
            other_cpu_s,
            grade,
            rank,
            idle_before_w: None,
            idle_after_w: None,
            baseline_drift_w: None,
            paired_delta_j: None,
            paired_attributable: false,
            other_core_busy: String::new(),
            cadence_s: f64::NAN,
            left_edge: String::new(),
            right_edge: String::new(),
            uncovered_s: 0.0,
            samples: counter_updates,
            quantization_window_j: None,
            cadence_edge_window_j: f64::NAN,
        }
    }
}

/// Uncertainty of joules per effective sample for one cell.
///
/// The Student-t half-width is the published scatter of the mean. The systematic terms are
/// the mean of the per-trial bounds and are not divided by the square root of the trial count:
/// a shared instrument limit does not shrink because the trial was repeated.
pub fn j_per_ess_uncertainty(trials: &[EnergyTrial]) -> UncertaintyBudget {
    let per_ess: Vec<f64> = trials.iter().filter_map(|t| t.joules_per_ess).collect();
    let student = mean_interval(&per_ess).map_or(f64::NAN, |(_, lo, hi, _)| 0.5 * (hi - lo));
    let mut quant = Vec::new();
    let mut edge = Vec::new();
    let mut bracket = Vec::new();
    for t in trials {
        if !(t.ess > 0.0) || !t.ess.is_finite() || t.repetitions == 0 {
            continue;
        }
        let reps = f64::from(t.repetitions);
        if let Some(q) = t.quantization_window_j {
            quant.push(q / reps / t.ess);
        }
        if t.cadence_edge_window_j.is_finite() {
            edge.push(t.cadence_edge_window_j / reps / t.ess);
        }
        if let Some(sig) = t.joules.idle_sigma.filter(|s| s.is_finite()) {
            bracket.push(sig * t.joules.seconds / t.ess);
        }
    }
    let mean = |v: &[f64]| if v.is_empty() { None } else { Some(v.iter().sum::<f64>() / v.len() as f64) };
    combine_uncertainty(student, mean(&quant), mean(&edge), mean(&bracket).unwrap_or(f64::NAN))
}

/// Whether two trials are the same cell of a table.
///
/// A cell is one workload, one sampler, one instrument, one scope, and one evidence grade.
/// Package joules and SoC-telemetry joules are not a cell.
pub fn may_share_summary(a: &EnergyTrial, b: &EnergyTrial) -> bool {
    a.workload == b.workload
        && a.side == b.side
        && a.joules.instrument == b.joules.instrument
        && scopes_may_share_a_table(&a.joules.scope, &b.joules.scope)
        && grades_may_share_a_table(a.grade, b.grade)
}

/// Intervals for one cell. Refuses a slice that mixes workload, sampler, scope, instrument, or grade.
pub fn summarise_one_grade(trials: &[EnergyTrial]) -> Result<GradeSummary, &'static str> {
    let Some(first) = trials.first() else {
        return Err("no trials");
    };
    if trials.iter().any(|t| !may_share_summary(t, first)) {
        return Err("mixed workload, sampler, scope, instrument, or evidence grade");
    }
    let gross: Vec<f64> = trials.iter().filter_map(|t| t.joules.gross).collect();
    let incremental: Vec<f64> = trials.iter().filter_map(|t| t.joules.incremental).collect();
    let per_ess: Vec<f64> = trials.iter().filter_map(|t| t.joules_per_ess).collect();
    Ok(GradeSummary {
        workload: first.workload,
        side: first.side,
        instrument: first.joules.instrument.clone(),
        scope: first.joules.scope.clone(),
        grade: first.grade,
        n: trials.len(),
        gross: mean_interval(&gross),
        incremental: mean_interval(&incremental),
        joules_per_ess: mean_interval(&per_ess),
        uncertainty: j_per_ess_uncertainty(trials),
    })
}

/// One summary per workload, sampler, instrument, and scope, in first-seen order.
///
/// Package rows and SoC-telemetry rows stay in different summaries.
pub fn summarise_workloads(trials: &[EnergyTrial]) -> Vec<GradeSummary> {
    let mut keys: Vec<(&str, &str, &str, &str, &str)> = Vec::new();
    for t in trials {
        let key = (t.workload, t.side, t.joules.instrument.as_str(), t.joules.scope.as_str(), t.grade);
        if !keys.contains(&key) {
            keys.push(key);
        }
    }
    keys.into_iter()
        .filter_map(|key| {
            let group: Vec<EnergyTrial> = trials
                .iter()
                .filter(|t| {
                    (t.workload, t.side, t.joules.instrument.as_str(), t.joules.scope.as_str(), t.grade) == key
                })
                .cloned()
                .collect();
            summarise_one_grade(&group).ok()
        })
        .collect()
}

pub struct GradeSummary {
    pub workload: &'static str,
    pub side: &'static str,
    pub instrument: String,
    pub scope: String,
    pub grade: &'static str,
    pub n: usize,
    pub gross: Option<(f64, f64, f64, usize)>,
    pub incremental: Option<(f64, f64, f64, usize)>,
    pub joules_per_ess: Option<(f64, f64, f64, usize)>,
    /// Combined uncertainty of joules per effective sample, beside the Student-t interval.
    pub uncertainty: UncertaintyBudget,
}

/// 95% Student-t interval over the finite values. `None` when fewer than two exist.
pub fn mean_interval(xs: &[f64]) -> Option<(f64, f64, f64, usize)> {
    crate::score::mean_ci_95(xs)
}

/// Joules per effective sample, only from a measurement that cleared the noise floor.
pub fn joules_per_ess(joules: &Joules, ess: f64) -> Option<f64> {
    if !joules.attributable || !(ess > 0.0) || !ess.is_finite() {
        return None;
    }
    joules.incremental.filter(|j| j.is_finite()).map(|j| j / ess)
}

pub struct Probe {
    pub load1: Option<f64>,
    pub quiet: bool,
    /// macmon or INA3221. A RAPL meter is not kept here: RAPL is read as a counter below.
    pub wall: Option<Meter>,
    pub wall_note: String,
    pub rapl: Option<RaplCounter>,
    pub rapl_note: String,
    /// The `core` subdomain (`intel-rapl:0:0`). Not added to the package counter.
    pub rapl_core: Option<RaplCounter>,
    pub rapl_core_note: String,
    pub perf: Option<PerfPkg>,
    pub perf_note: String,
    pub nvml: Option<Nvml>,
    pub nvml_note: String,
    /// Jetson `tegrastats` `VDD_IN`, only when no wall backend opened. Not a sum of rails.
    pub tegra: Option<Tegra>,
    pub tegra_note: String,
    pub hostname: String,
    pub macos_policy: &'static str,
}

impl Probe {
    pub fn detect() -> Probe {
        let load1 = load_average();
        let quiet = load1.is_some_and(|l| l <= QUIET_LOAD);
        let wall = Meter::detect().filter(|m| m.backend() != "rapl");
        let wall_note = match &wall {
            Some(m) => format!("backend={} machine={} scope={}", m.backend(), m.machine(), m.scope().label()),
            None => "no macmon and no INA3221 rail".into(),
        };
        let (rapl, rapl_note) = match RaplCounter::open() {
            Ok(c) => {
                let note = format!("{} ({})", c.name, c.kind);
                (Some(c), note)
            }
            Err(e) => (None, e),
        };
        let (rapl_core, rapl_core_note) = match rapl.as_ref().and_then(|c| RaplCounter::open_core(c.counter_path())) {
            Some(c) => (Some(c), "core subdomain, reported separately from package".into()),
            None => (None, "no readable core subdomain".into()),
        };
        let (perf, perf_note) = if rapl.is_some() {
            (None, "not opened; RAPL powercap is the CPU counter".into())
        } else {
            match PerfPkg::open() {
                Ok(p) => (Some(p), "power/energy-pkg".into()),
                Err(e) => (None, e),
            }
        };
        let (nvml, nvml_note) = match Nvml::open() {
            Ok(n) => {
                let note = format!(
                    "{} device(s): {} via nvmlDeviceGetTotalEnergyConsumption",
                    n.devices.len(),
                    n.names.join(", ")
                );
                (Some(n), note)
            }
            Err(e) => (None, e),
        };
        let (tegra, tegra_note) = if wall.is_some() {
            (None, "not opened; a power backend (macmon SoC telemetry or INA3221) is already open".into())
        } else {
            match Tegra::open() {
                Ok(t) => (Some(t), "tegrastats VDD_IN, rootless, other rails not summed".into()),
                Err(e) => (None, e),
            }
        };
        Probe {
            load1,
            quiet,
            wall,
            wall_note,
            rapl,
            rapl_note,
            rapl_core,
            rapl_core_note,
            perf,
            perf_note,
            nvml,
            nvml_note,
            tegra,
            tegra_note,
            hostname: read_hostname(),
            macos_policy: "macOS power, when that OS is the host, is ferrotherm_meter::Meter::macmon \
                           (rootless IOReport via the macmon binary). This harness has no other macOS path.",
        }
    }

}

/// One `tegrastats` line's `VDD_IN` rail, in watts.
///
/// The token is `1864/1864` or `1864mW/1864mW`. The first number is the instantaneous
/// draw, in milliwatts. A line that only names `VDD_CPU_GPU_CV` or `VDD_SOC` is `None`:
/// those rails are inside `VDD_IN`, and summing them would count the same current twice.
pub fn parse_tegrastats_vdd_in_watts(line: &str) -> Option<f64> {
    let mut tokens = line.split_whitespace();
    while let Some(tok) = tokens.next() {
        if tok == "VDD_IN" {
            let raw = tokens.next()?;
            let head = raw.split('/').next().unwrap_or(raw);
            let digits: String = head.chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
            let mw: f64 = digits.parse().ok()?;
            if mw.is_finite() && mw >= 0.0 {
                return Some(mw / 1000.0);
            }
            return None;
        }
    }
    None
}

/// A rootless `tegrastats` reader. The child is stopped when this value is dropped.
pub struct Tegra {
    child: std::process::Child,
    samples: std::sync::Arc<std::sync::Mutex<Vec<(std::time::Instant, f64)>>>,
}

impl Tegra {
    pub fn open() -> Result<Tegra, String> {
        let mut child = std::process::Command::new("tegrastats")
            .args(["--interval", "200"])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| format!("tegrastats was not started: {e}"))?;
        let stdout = child.stdout.take().ok_or("tegrastats has no stdout")?;
        let samples = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = std::sync::Arc::clone(&samples);
        std::thread::spawn(move || {
            let mut reader = std::io::BufReader::new(stdout);
            let mut line = String::new();
            loop {
                line.clear();
                let n = std::io::BufRead::read_line(&mut reader, &mut line).unwrap_or(0);
                if n == 0 {
                    break;
                }
                if let Some(w) = parse_tegrastats_vdd_in_watts(&line) {
                    sink.lock().unwrap().push((std::time::Instant::now(), w));
                }
            }
        });
        let start = std::time::Instant::now();
        while start.elapsed() < Duration::from_secs(2) {
            if !samples.lock().unwrap().is_empty() {
                return Ok(Tegra { child, samples });
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = child.kill();
        Err("tegrastats produced no VDD_IN sample. Other rails were not summed.".into())
    }

    /// Samples at or after `from`, plus the last sample before `from` when one exists.
    ///
    /// The earlier sample is the left bracket for a trapezoid. Dropping it would leave the
    /// start of the window uncovered.
    pub fn since_stamped(&self, from: std::time::Instant) -> Vec<(std::time::Instant, f64)> {
        let v = self.samples.lock().unwrap();
        let mut out = Vec::new();
        if let Some(prev) = v.iter().rev().find(|(t, _)| *t < from) {
            out.push(*prev);
        }
        out.extend(v.iter().copied().filter(|(t, _)| *t >= from));
        out
    }
}

impl Drop for Tegra {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn read_hostname() -> String {
    std::fs::read_to_string("/etc/hostname")
        .or_else(|_| std::fs::read_to_string("/proc/sys/kernel/hostname"))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

/// A RAPL energy counter. Package-only when `psys` does not contain the package.
pub struct RaplCounter {
    path: std::path::PathBuf,
    max_uj: u64,
    pub name: String,
    pub kind: &'static str,
}

impl RaplCounter {
    pub fn open() -> Result<RaplCounter, String> {
        let root = std::path::Path::new("/sys/class/powercap");
        let rd = match std::fs::read_dir(root) {
            Ok(rd) => rd,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err("RAPL powercap is not present on this machine".into());
            }
            Err(e) => return Err(format!("RAPL powercap: {e}")),
        };
        let mut denied = Vec::new();
        let mut domains = Vec::new();
        for ent in rd.flatten() {
            let p = ent.path();
            let base = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if base.matches(':').count() != 1 {
                continue;
            }
            let name = std::fs::read_to_string(p.join("name")).unwrap_or_default();
            let name = name.trim().to_string();
            match std::fs::read_to_string(p.join("energy_uj")) {
                Ok(_) => {
                    let max = std::fs::read_to_string(p.join("max_energy_range_uj"))
                        .ok()
                        .and_then(|s| s.trim().parse().ok())
                        .unwrap_or(u64::MAX);
                    domains.push((p, name, max));
                }
                Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                    denied.push(format!("{base} ({name})"));
                }
                Err(_) => {}
            }
        }
        if domains.is_empty() {
            if denied.is_empty() {
                return Err("RAPL powercap has no readable top-level energy_uj".into());
            }
            return Err(format!(
                "RAPL energy_uj permission denied: {}. No privilege was requested.",
                denied.join(", ")
            ));
        }
        let pkg = domains.iter().position(|(_, n, _)| n.starts_with("package"));
        let psys = domains.iter().position(|(_, n, _)| n == "psys");
        let Some(pkg_i) = pkg else {
            return Err("RAPL exposed no package domain".into());
        };
        let use_psys = if let Some(i) = psys {
            let pw = watts_over(&domains[i].0, 0.3);
            let kw = watts_over(&domains[pkg_i].0, 0.3);
            choose_rapl_kind(pw, kw) == Some("psys")
        } else {
            false
        };
        let idx = if use_psys { psys.expect("psys") } else { pkg_i };
        let (path, name, max_uj) = domains.swap_remove(idx);
        let kind = if use_psys { "psys" } else { "package" };
        Ok(RaplCounter { path, max_uj, name, kind })
    }

    pub fn counter_path(&self) -> &std::path::Path {
        &self.path
    }

    pub fn range_j(&self) -> f64 {
        self.max_uj as f64 / 1e6
    }

    pub fn read_uj(&self) -> Result<u64, String> {
        std::fs::read_to_string(self.path.join("energy_uj"))
            .map_err(|e| format!("RAPL read: {e}"))?
            .trim()
            .parse()
            .map_err(|e| format!("RAPL parse: {e}"))
    }

    pub fn delta_j(&self, prev: u64, now: u64) -> f64 {
        counter_delta_j(prev as f64 / 1e6, now as f64 / 1e6, self.max_uj as f64 / 1e6)
    }

    /// The `core` child of a package domain, if its `energy_uj` is readable.
    ///
    /// This is `intel-rapl:0:0` on the Framework machine. It is a subdomain of the package,
    /// so its joules are not added to the package joules.
    pub fn open_core(package: &std::path::Path) -> Option<RaplCounter> {
        let rd = std::fs::read_dir(package).ok()?;
        for ent in rd.flatten() {
            let p = ent.path();
            let name = std::fs::read_to_string(p.join("name")).unwrap_or_default();
            if name.trim() != "core" {
                continue;
            }
            if std::fs::read_to_string(p.join("energy_uj")).is_err() {
                continue;
            }
            let max = std::fs::read_to_string(p.join("max_energy_range_uj"))
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(u64::MAX);
            return Some(RaplCounter { path: p, max_uj: max, name: "core".into(), kind: "core" });
        }
        None
    }
}

fn watts_over(dir: &std::path::Path, secs: f64) -> Option<f64> {
    let read = |p: &std::path::Path| -> Option<u64> {
        std::fs::read_to_string(p.join("energy_uj")).ok()?.trim().parse().ok()
    };
    let a = read(dir)?;
    let t = std::time::Instant::now();
    std::thread::sleep(Duration::from_secs_f64(secs));
    let b = read(dir)?;
    let max: u64 = std::fs::read_to_string(dir.join("max_energy_range_uj")).ok()?.trim().parse().ok()?;
    let d = if b >= a { b - a } else { max.saturating_sub(a).saturating_add(b) };
    Some((d as f64 / 1e6) / t.elapsed().as_secs_f64())
}

/// `power/energy-pkg` via `perf_event_open`. Failure text includes `perf stat` when the syscall misses.
pub struct PerfPkg {
    fd: i32,
    scale: f64,
}

impl PerfPkg {
    pub fn open() -> Result<PerfPkg, String> {
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            return Err("perf energy-pkg is not opened on this architecture".into());
        }
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            open_perf_pkg()
        }
    }

    pub fn read_raw(&self) -> Result<u64, String> {
        let mut buf = [0u8; 8];
        let n = unsafe { read(self.fd, buf.as_mut_ptr().cast(), buf.len()) };
        if n != 8 {
            return Err(format!("perf energy-pkg read returned {n}"));
        }
        Ok(u64::from_ne_bytes(buf))
    }

    pub fn read_j(&self) -> Result<f64, String> {
        Ok(self.read_raw()? as f64 * self.scale)
    }

    pub fn fd_and_scale(&self) -> (i32, f64) {
        (self.fd, self.scale)
    }
}

pub fn perf_sample_j(fd: i32, scale: f64) -> Option<f64> {
    let mut buf = [0u8; 8];
    let n = unsafe { read(fd, buf.as_mut_ptr().cast(), buf.len()) };
    if n == 8 { Some(u64::from_ne_bytes(buf) as f64 * scale) } else { None }
}

impl Drop for PerfPkg {
    fn drop(&mut self) {
        unsafe { close(self.fd) };
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn open_perf_pkg() -> Result<PerfPkg, String> {
    let ty_s = std::fs::read_to_string("/sys/bus/event_source/devices/power/type")
        .map_err(|e| format!("power PMU type: {e}. {}", perf_stat_note()))?;
    let ev = std::fs::read_to_string("/sys/bus/event_source/devices/power/events/energy-pkg")
        .map_err(|e| format!("energy-pkg event: {e}. {}", perf_stat_note()))?;
    let scale = std::fs::read_to_string("/sys/bus/event_source/devices/power/events/energy-pkg.scale")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(1.0);
    let ty: u32 = match ty_s.trim().parse() {
        Ok(v) => v,
        Err(_) => return Err(format!("bad power PMU type {ty_s}")),
    };
    let config = parse_perf_event(&ev)?;
    let mut attr = [0u8; 64];
    attr[0..4].copy_from_slice(&ty.to_ne_bytes());
    attr[4..8].copy_from_slice(&64u32.to_ne_bytes());
    attr[8..16].copy_from_slice(&config.to_ne_bytes());
    let fd = unsafe {
        syscall(
            SYS_PERF_EVENT_OPEN,
            attr.as_ptr(),
            -1isize,
            0isize,
            -1isize,
            0u64,
        )
    };
    if fd < 0 {
        let err = std::io::Error::last_os_error();
        return Err(format!("perf_event_open power/energy-pkg: {err}. {}", perf_stat_note()));
    }
    Ok(PerfPkg { fd: fd as i32, scale })
}

fn perf_stat_note() -> String {
    match std::process::Command::new("perf").args(["stat", "-e", "power/energy-pkg/", "--", "true"]).output() {
        Ok(o) => {
            let err = String::from_utf8_lossy(&o.stderr);
            let err = err.trim();
            let parsed = match parse_perf_joules(err) {
                Some(j) => format!("; that text also parsed as {j} J, which is not a trial"),
                None => String::new(),
            };
            if err.is_empty() {
                format!("perf stat exit {}{parsed}", o.status)
            } else {
                format!("perf stat: {err}{parsed}")
            }
        }
        Err(e) => format!("perf stat was not started: {e}"),
    }
}

#[cfg(target_arch = "x86_64")]
const SYS_PERF_EVENT_OPEN: i64 = 298;
#[cfg(target_arch = "aarch64")]
const SYS_PERF_EVENT_OPEN: i64 = 241;

unsafe extern "C" {
    fn syscall(n: i64, attr: *const u8, pid: isize, cpu: isize, group: isize, flags: u64) -> i64;
    // POSIX `read` as the standard library names it. `*mut u8` is the same pointer
    // ABI and is the declaration rustc 1.99 warned as a mismatched runtime symbol.
    fn read(fd: i32, buf: *mut std::ffi::c_void, count: usize) -> isize;
    fn close(fd: i32) -> i32;
}

/// `sched_setaffinity(0, mask)` without a second declaration of `syscall`.
fn affinity_syscall(mask: &[u8]) -> i64 {
    let mut rc: i64;
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::asm!(
            "syscall",
            in("rax") 203i64,
            in("rdi") 0i64,
            in("rsi") mask.len(),
            in("rdx") mask.as_ptr(),
            lateout("rax") rc,
            out("rcx") _,
            out("r11") _,
        );
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!(
            "svc #0",
            in("x8") 122i64,
            in("x0") 0i64,
            in("x1") mask.len(),
            in("x2") mask.as_ptr(),
            lateout("x0") rc,
        );
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        rc = -1;
        let _ = mask;
    }
    rc
}

/// Pin this process to `cpus`. Children spawned afterwards inherit the mask.
///
/// `sched_setaffinity` is tried first. If it fails, `taskset -c LIST -p PID` is tried.
/// Neither path asks for a privilege the process does not already have.
pub fn pin_current(cpus: &[usize]) -> Result<(), String> {
    if cpus.is_empty() {
        return Err("cpu pin was empty".into());
    }
    let max = *cpus.iter().max().expect("nonempty");
    let nbytes = (max / 8 + 1).next_multiple_of(8);
    let mut mask = vec![0u8; nbytes];
    for &c in cpus {
        mask[c / 8] |= 1 << (c % 8);
    }
    let rc = affinity_syscall(&mask);
    if rc == 0 {
        return Ok(());
    }
    let list = cpus.iter().map(std::string::ToString::to_string).collect::<Vec<_>>().join(",");
    let pid = std::process::id().to_string();
    match std::process::Command::new("taskset").args(["-c", &list, "-p", &pid]).status() {
        Ok(s) if s.success() => Ok(()),
        Ok(s) => Err(format!("sched_setaffinity returned {rc} and taskset exited {s}")),
        Err(e) => Err(format!("sched_setaffinity returned {rc} and taskset was not started: {e}")),
    }
}

/// Snapshot of machine busy ticks and this process's own ticks.
pub struct CpuSnap {
    pub busy: u64,
    pub mine: u64,
    pub load1: Option<f64>,
}

pub fn read_cpu_snap() -> CpuSnap {
    let busy = std::fs::read_to_string("/proc/stat")
        .ok()
        .and_then(|s| s.lines().find_map(parse_proc_stat_busy))
        .unwrap_or(0);
    let mine = std::fs::read_to_string("/proc/self/stat").ok().and_then(|s| parse_self_stat_busy(&s)).unwrap_or(0);
    CpuSnap { busy, mine, load1: load_average() }
}

pub fn snap_other_cpu(before: &CpuSnap, after: &CpuSnap) -> Option<f64> {
    if before.busy == 0 && after.busy == 0 {
        return None;
    }
    Some(other_cpu_seconds(before.busy, before.mine, after.busy, after.mine, 100.0))
}

/// One device's NVML energy counter, in millijoules since driver load.
pub struct NvmlDevice {
    handle: *mut std::ffi::c_void,
    pub name: String,
}

pub struct Nvml {
    lib: *mut std::ffi::c_void,
    get_energy: GetEnergy,
    devices: Vec<NvmlDevice>,
    pub names: Vec<String>,
}

type GetEnergy = unsafe extern "C" fn(*mut std::ffi::c_void, *mut u64) -> i32;

impl Nvml {
    /// Load `libnvidia-ml.so.1` and the total-energy-consumption counter. `Err` when the
    /// library, the symbol, or a device is absent. Never a zero that stands in for a missing card.
    pub fn open() -> Result<Nvml, String> {
        #[cfg(not(unix))]
        {
            return Err("NVML is opened with dlopen on unix only".into());
        }
        #[cfg(unix)]
        {
            open_unix()
        }
    }

    /// Sum of per-device counter readings, in joules. Devices are summed and the sum is
    /// labelled as a sum of GPU counters, not as whole-machine energy.
    pub fn millijoules(&self) -> Result<u64, String> {
        let mut total_mj = 0u64;
        for d in &self.devices {
            let mut mj = 0u64;
            let rc = unsafe { (self.get_energy)(d.handle, &mut mj) };
            if rc != 0 {
                return Err(format!("NVML energy read on {} returned {rc}", d.name));
            }
            total_mj = total_mj.saturating_add(mj);
        }
        Ok(total_mj)
    }

    pub fn joules(&self) -> Result<f64, String> {
        Ok(self.millijoules()? as f64 / 1000.0)
    }

    /// A copy of the handles the cadence poller can own.
    ///
    /// The poller is the only caller of these handles while a window is running.
    /// `nvmlDeviceGetTotalEnergyConsumption` is safe to call from that thread for the
    /// duration of the window; the main thread does not call it then.
    pub fn sampler(&self) -> NvmlSampler {
        NvmlSampler {
            get_energy: self.get_energy,
            handles: self.devices.iter().map(|d| d.handle).collect(),
        }
    }
}

pub struct NvmlSampler {
    get_energy: GetEnergy,
    handles: Vec<*mut std::ffi::c_void>,
}

// The poller thread is the sole user of these device handles for the window.
unsafe impl Send for NvmlSampler {}

impl NvmlSampler {
    pub fn millijoules(&self) -> Option<u64> {
        let mut total = 0u64;
        for handle in &self.handles {
            let mut mj = 0u64;
            if unsafe { (self.get_energy)(*handle, &mut mj) } != 0 {
                return None;
            }
            total = total.saturating_add(mj);
        }
        Some(total)
    }
}

impl Drop for Nvml {
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            if !self.lib.is_null() {
                dlclose(self.lib);
            }
        }
    }
}

#[cfg(unix)]
fn open_unix() -> Result<Nvml, String> {
    // SAFETY: `libnvidia-ml.so.1` is loaded by name, every symbol is checked for NULL
    // before it is transmuted to the C function it documents, and a failed load is an
    // `Err` rather than a null call. The handles are only used through those functions.
    let path = std::ffi::CString::new("libnvidia-ml.so.1").expect("static path");
    let lib = unsafe { dlopen(path.as_ptr(), 2) }; // RTLD_NOW
    if lib.is_null() {
        return Err("libnvidia-ml.so.1 did not load (no NVIDIA Management Library on this machine)".into());
    }
    let sym = |name: &str| -> Result<*mut std::ffi::c_void, String> {
        let c = std::ffi::CString::new(name).expect("symbol");
        let p = unsafe { dlsym(lib, c.as_ptr()) };
        if p.is_null() {
            unsafe { dlclose(lib) };
            return Err(format!("NVML loaded but {name} is missing"));
        }
        Ok(p)
    };
    type Init = unsafe extern "C" fn() -> i32;
    type Count = unsafe extern "C" fn(*mut u32) -> i32;
    type ByIndex = unsafe extern "C" fn(u32, *mut *mut std::ffi::c_void) -> i32;
    type GetName = unsafe extern "C" fn(*mut std::ffi::c_void, *mut i8, u32) -> i32;
    let init: Init = unsafe { std::mem::transmute(sym("nvmlInit_v2")?) };
    let count_fn: Count = unsafe { std::mem::transmute(sym("nvmlDeviceGetCount_v2")?) };
    let by_index: ByIndex = unsafe { std::mem::transmute(sym("nvmlDeviceGetHandleByIndex_v2")?) };
    let get_name: GetName = unsafe { std::mem::transmute(sym("nvmlDeviceGetName")?) };
    let get_energy: GetEnergy = unsafe { std::mem::transmute(sym("nvmlDeviceGetTotalEnergyConsumption")?) };
    if unsafe { init() } != 0 {
        unsafe { dlclose(lib) };
        return Err("nvmlInit_v2 failed".into());
    }
    let mut n = 0u32;
    if unsafe { count_fn(&mut n) } != 0 || n == 0 {
        unsafe { dlclose(lib) };
        return Err("NVML initialised and reported no device".into());
    }
    let mut devices = Vec::new();
    let mut names = Vec::new();
    for i in 0..n {
        let mut handle: *mut std::ffi::c_void = std::ptr::null_mut();
        if unsafe { by_index(i, &mut handle) } != 0 {
            unsafe { dlclose(lib) };
            return Err(format!("NVML device {i} has no handle"));
        }
        let mut buf = [0i8; 96];
        let name = if unsafe { get_name(handle, buf.as_mut_ptr(), buf.len() as u32) } == 0 {
            let bytes: Vec<u8> = buf.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
            String::from_utf8_lossy(&bytes).into_owned()
        } else {
            format!("gpu{i}")
        };
        names.push(name.clone());
        devices.push(NvmlDevice { handle, name });
    }
    Ok(Nvml { lib, get_energy, devices, names })
}

#[cfg(unix)]
unsafe extern "C" {
    fn dlopen(filename: *const i8, flags: i32) -> *mut std::ffi::c_void;
    fn dlsym(handle: *mut std::ffi::c_void, symbol: *const i8) -> *mut std::ffi::c_void;
    fn dlclose(handle: *mut std::ffi::c_void) -> i32;
}

/// Millijoule counter delta, wrap included, as joules.
pub fn mj_delta_j(prev: u64, now: u64) -> f64 {
    let d = if now >= prev { now - prev } else { now.wrapping_add(u64::MAX - prev).wrapping_add(1) };
    d as f64 / 1000.0
}

/// Joules across a series of millijoule readings, using [`mj_delta_j`] for the wrap.
pub fn mj_window_j(readings_mj: &[f64]) -> f64 {
    let Some(first) = readings_mj.iter().copied().find(|r| r.is_finite()) else {
        return f64::NAN;
    };
    let Some(last) = readings_mj.iter().copied().rev().find(|r| r.is_finite()) else {
        return f64::NAN;
    };
    mj_delta_j(first as u64, last as u64)
}

/// Quiet idle on a wall-power meter. The meter's own `idle` refuses a load above [`QUIET_LOAD`].
/// A contended baseline is [`ferrotherm_meter::Meter::observe_idle`], and that result is a
/// different evidence grade.
pub fn idle_baseline(meter: &mut ferrotherm_meter::Meter) -> Result<ferrotherm_meter::Baseline, String> {
    meter.idle(Duration::from_secs(2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incremental_is_gross_minus_idle_and_the_band_is_visible() {
        let (inc, ok) = split_incremental(10.0, 4.0, 0.1, 2.0);
        assert!((inc - 2.0).abs() < 1e-12);
        assert!(ok, "2 J clears 3 * 0.1 W * 2 s = 0.6 J");
        let (small, ok) = split_incremental(8.1, 4.0, 0.1, 2.0);
        assert!((small - 0.1).abs() < 1e-12);
        assert!(!ok, "0.1 J does not clear a 0.6 J band");
        let (neg, ok) = split_incremental(4.0, 4.0, 0.1, 2.0);
        assert!((neg + 4.0).abs() < 1e-12);
        assert!(!ok, "a negative incremental is not attributable even when it clears the band");
    }

    #[test]
    fn a_missing_counter_does_not_become_zero_joules() {
        let j = Joules::unavailable(1.0, "none", "no counter");
        assert!(j.gross.is_none());
        assert!(j.incremental.is_none());
        assert!(!j.attributable);
        assert!(!j.projected_prices_applied);
        assert!(joules_per_ess(&j, 10.0).is_none());
    }

    #[test]
    fn an_unattributable_delta_is_not_a_per_sample_price() {
        let j = Joules {
            gross: Some(8.1),
            incremental: Some(0.1),
            idle_watts: Some(4.0),
            idle_sigma: Some(0.1),
            seconds: 2.0,
            attributable: false,
            evidence: "metered",
            instrument: "rapl".into(),
            scope: "package".into(),
            note: String::new(),
            projected_prices_applied: false,
        };
        assert!(joules_per_ess(&j, 100.0).is_none());
    }

    #[test]
    fn the_quiet_gate_is_the_host_constant_and_is_not_loosened() {
        assert_eq!(QUIET_LOAD, ferrotherm::host::QUIET_LOAD);
        assert_eq!(QUIET_LOAD, 2.0);
        let at = poll_quiet(&[Some(2.0)]);
        assert_eq!(at.outcome, "quiet");
        let over = poll_quiet(&[Some(2.01)]);
        assert_eq!(over.outcome, "load_gated");
        let mac = poll_quiet(&[Some(6.45), Some(6.45)]);
        assert_eq!(mac.outcome, "load_gated");
        assert!(mac.accepted.is_none());
        assert_eq!(mac.samples, vec![6.45, 6.45]);
    }

    #[test]
    fn a_later_quiet_sample_ends_the_wait_and_a_stuck_host_stays_gated() {
        let ok = poll_quiet(&[Some(6.45), Some(4.0), Some(1.2)]);
        assert_eq!(ok.outcome, "quiet");
        assert_eq!(ok.accepted, Some(1.2));
        assert_eq!(ok.samples.len(), 3);
        let stuck = poll_quiet(&[Some(3.0); 15]);
        assert_eq!(stuck.outcome, "load_gated");
        assert_eq!(stuck.samples.len(), 15);
    }

    #[test]
    fn contended_mode_is_never_graded_quiet() {
        assert_eq!(grade_for(true, true), GRADE_CONTENDED);
        assert_eq!(grade_for(true, false), GRADE_CONTENDED);
        assert_eq!(grade_for(false, true), GRADE_QUIET);
        assert_eq!(grade_for(false, false), "load_gated");
        assert!(!grades_may_share_a_table(GRADE_QUIET, GRADE_CONTENDED));
        assert!(grades_may_share_a_table(GRADE_QUIET, GRADE_QUIET));
        assert!(grades_may_share_a_table(GRADE_CONTENDED, GRADE_CONTENDED));
    }

    #[test]
    fn rapl_on_the_named_framework_is_primary_and_macmon_is_secondary() {
        assert_eq!(energy_rank("rapl", "framework-128gb-0000"), "primary");
        assert_eq!(energy_rank("rapl", "framework-32gb-0001"), "unranked");
        assert_eq!(energy_rank("macmon", "Davids-MacBook-Pro-M5-Max-48GB-4.local"), "secondary");
        assert_eq!(energy_rank("perf", "framework-128gb-0000"), "unranked");
    }

    #[test]
    fn interleaved_order_starts_aba_and_gives_each_side_five() {
        let mut seq = Vec::new();
        for pair in 0..5 {
            seq.extend(pair_order(pair));
        }
        assert_eq!(&seq[..4], &["ferrotherm", "thrml", "thrml", "ferrotherm"]);
        assert_eq!(seq.iter().filter(|s| **s == "ferrotherm").count(), 5);
        assert_eq!(seq.iter().filter(|s| **s == "thrml").count(), 5);
    }

    #[test]
    fn other_process_cpu_is_the_machine_minus_this_process() {
        // 300 machine ticks, 50 of them ours, at 100 Hz → 2.5 s of everyone else.
        let s = other_cpu_seconds(1_000, 100, 1_300, 150, 100.0);
        assert!((s - 2.5).abs() < 1e-12);
        assert_eq!(other_cpu_seconds(50, 10, 40, 10, 100.0), 0.0);
        let line = "cpu  10 1 2 100 0 3 4 5 0 0";
        assert_eq!(parse_proc_stat_busy(line), Some(10 + 1 + 2 + 3 + 4 + 5));
        let me = "9 (prog) S 1 1 1 0 0 0 0 0 0 0 7 8";
        assert_eq!(parse_self_stat_busy(me), Some(15));
    }

    #[test]
    fn incremental_ci_stays_inside_one_grade() {
        let mk = |g: &'static str, inc: f64| {
            let j = trial_joules(inc + 10.0, 10.0, 0.01, 1.0, "rapl", "package", g, MIN_COUNTER_UPDATES, true);
            EnergyTrial::from_parts(
                "ising_grid2d_open_3x3",
                "ferrotherm",
                0,
                1,
                MIN_COUNTER_UPDATES,
                1.0,
                j,
                100.0,
                Some(0.4),
                Some(0.1),
                g,
                "primary",
            )
        };
        let quiet: Vec<_> = (0..5).map(|i| mk(GRADE_QUIET, 1.0 + i as f64 * 0.1)).collect();
        let sum = summarise_one_grade(&quiet).unwrap();
        assert_eq!(sum.grade, GRADE_QUIET);
        assert!(sum.incremental.is_some());
        assert!(sum.joules_per_ess.is_some());
        let mut mixed = quiet;
        mixed.push(mk(GRADE_CONTENDED, 1.0));
        assert!(summarise_one_grade(&mixed).is_err());
    }

    #[test]
    fn psys_below_the_package_is_not_the_domain() {
        assert_eq!(choose_rapl_kind(Some(0.2), Some(3.3)), Some("package"));
        assert_eq!(choose_rapl_kind(Some(10.0), Some(3.3)), Some("psys"));
        assert_eq!(parse_perf_event("event=0x02").unwrap(), 2);
        assert_eq!(parse_perf_event("event=0x01,umask=0x10").unwrap(), 0x1001);
        let line = "0.50;Joules;power/energy-pkg;\n";
        assert!((parse_perf_joules(line).unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn a_counter_wrap_is_not_negative_energy() {
        // 100 mJ, then a wrap to 50 mJ: 2^64 - 100 + 50 mJ, reported in joules.
        let d = mj_delta_j(100, 50);
        assert!(d > 0.0);
        assert!((mj_delta_j(0, 2500) - 2.5).abs() < 1e-12);
        assert!((counter_delta_j(0.9, 0.1, 1.0) - 0.2).abs() < 1e-12);
        assert!((counter_delta_j(1.0, 1.4, 10.0) - 0.4).abs() < 1e-12);
        let windows = [(1.0, 0.5), (2.0, 0.5)];
        let (mean, sigma) = watts_from_windows(&windows).unwrap();
        assert!((mean - 3.0).abs() < 1e-12);
        assert!(sigma.is_finite());
        let gated = load_gated("busy");
        assert!(gated.gross.is_none());
        assert_eq!(gated.evidence, "load_gated");
        assert!(!gated.projected_prices_applied);
    }

    #[test]
    fn a_short_window_is_divided_by_repetitions_and_a_missed_cadence_is_not_attributable() {
        let (j, s) = per_repetition(90.0, 2.0, 2000);
        assert!((j - 0.045).abs() < 1e-12);
        assert!((s - 0.001).abs() < 1e-12);
        assert!(per_repetition(1.0, 2.0, 0).0.is_nan());
        assert_eq!(count_counter_updates(&[10.0, 10.0, 10.0]), 0);
        assert_eq!(count_counter_updates(&[10.0, 10.0, 10.4]), 1);
        assert_eq!(count_counter_updates(&[1.0, 2.0, 2.0, 3.0, f64::NAN, 4.0]), 3);
        assert!(count_counter_updates(&[0.0; 9]) < MIN_COUNTER_UPDATES);

        // 189502d, chain workload, ferrotherm trial 3: 0.827 ms, gross 0, incremental about -0.031 J,
        // and the harness of that revision marked it attributable.
        let missed = trial_joules(
            0.0,
            37.4752469,
            0.200209929,
            0.000827,
            "rapl",
            "package",
            GRADE_CONTENDED,
            0,
            true,
        );
        let inc = missed.incremental.unwrap();
        assert!(inc < -0.03 && inc > -0.032, "{inc}");
        assert!(!missed.attributable);
        assert!(joules_per_ess(&missed, 1727.79274).is_none());

        let one_tick = trial_joules(
            0.045545,
            37.4752469,
            0.200209929,
            0.001082,
            "rapl",
            "package",
            GRADE_CONTENDED,
            1,
            true,
        );
        assert!(!one_tick.attributable, "one counter tick is below the cadence floor");

        let held = trial_joules(
            0.045,
            37.4752469,
            0.200209929,
            0.001,
            "rapl",
            "package",
            GRADE_CONTENDED,
            MIN_COUNTER_UPDATES,
            true,
        );
        assert!(held.attributable);
        assert!(joules_per_ess(&held, 1000.0).is_some());
    }

    #[test]
    fn package_core_and_soc_telemetry_are_not_one_table() {
        assert!(!scopes_may_share_a_table("package", "soc-telemetry"));
        assert!(!scopes_may_share_a_table("package", "core"));
        assert!(!scopes_may_share_a_table("core", "soc-telemetry"));
        assert!(scopes_may_share_a_table("package", "package"));
        assert!(scopes_may_share_a_table("soc-telemetry", "soc-telemetry"));
        assert!(!scopes_may_share_a_table("cpu-package", "cpu-package"));
        let row = |workload: &'static str, scope: &'static str| {
            let j = trial_joules(2.0, 1.0, 0.01, 1.0, if scope == "package" { "rapl" } else { "macmon" }, scope, GRADE_CONTENDED, MIN_COUNTER_UPDATES, true);
            EnergyTrial::from_parts(workload, "ferrotherm", 0, 4, MIN_COUNTER_UPDATES, 2.0, j, 10.0, Some(2.5), Some(0.2), GRADE_CONTENDED, "primary")
        };
        let package = row("thrml_readme_ising_chain_n5", "package");
        let soc = row("thrml_readme_ising_chain_n5", "soc-telemetry");
        let core = row("thrml_readme_ising_chain_n5", "core");
        let other = row("ising_grid2d_open_3x3", "package");
        assert!(!may_share_summary(&package, &soc));
        assert!(!may_share_summary(&package, &core));
        assert!(!may_share_summary(&package, &other));
        assert!(summarise_one_grade(&[package.clone(), soc.clone()]).is_err());
        assert!(summarise_one_grade(&[package.clone(), other.clone()]).is_err());
        let sums = summarise_workloads(&[package, soc, core, other]);
        assert_eq!(sums.len(), 4);
        assert_eq!(sums[0].scope, "package");
        assert_eq!(sums[0].workload, "thrml_readme_ising_chain_n5");
        assert_eq!(sums[1].scope, "soc-telemetry");
        assert_eq!(sums[1].instrument, "macmon");
        assert_eq!(sums[2].scope, "core");
        assert_eq!(sums[3].workload, "ising_grid2d_open_3x3");
        assert!(!scopes_may_share_a_table("gpu", "package"));
        assert!(!scopes_may_share_a_table("board", "soc-telemetry"));
        assert!(!scopes_may_share_a_table("gpu", "board"));
        assert!(!scopes_may_share_a_table("whole-system", "soc-telemetry"));
    }

    #[test]
    fn uncertainty_keeps_a_missing_quantum_out_of_the_root_sum_square() {
        let budget = combine_uncertainty(3.0, None, Some(4.0), 0.0);
        assert!((budget.combined - 5.0).abs() < 1e-12, "{}", budget.combined);
        assert!(budget.quantization.is_none());
        let stated = combine_uncertainty(0.0, Some(3.0), Some(4.0), 0.0);
        assert!((stated.combined - 5.0).abs() < 1e-12);
        assert_eq!(counter_quantum_j("rapl"), Some(1e-6));
        assert_eq!(counter_quantum_j("macmon"), None);
        assert!((counter_quantization_j(1e-6) - 2e-6).abs() < 1e-18);
        assert!((counter_cadence_edge_j(10.0, 2.0, 0.1) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn quiet_gate_holds_only_when_every_load_sample_is_at_or_under_the_threshold() {
        assert_eq!(QUIET_GATE_SECONDS, 60);
        let quiet = quiet_gate_from(&[Some(1.0), Some(QUIET_LOAD), Some(0.5)], &[(1, 0.1)], true, 60.0);
        assert!(quiet.held);
        assert_eq!(quiet.max_load1, Some(QUIET_LOAD));
        assert_eq!(quiet.other_core_busy, vec![(1, 0.1)]);
        let over = quiet_gate_from(&[Some(1.0), Some(QUIET_LOAD + 0.01)], &[], true, 60.0);
        assert!(!over.held, "one sample over the unchanged gate refuses the run");
        let missing = quiet_gate_from(&[Some(1.0), None], &[], false, 60.0);
        assert!(!missing.held);
        assert!(!missing.busy_readable);
        assert!(!quiet_gate_from(&[], &[], true, 0.0).held);
    }

    #[test]
    fn a_held_quiet_gate_runs_energy_before_accuracy() {
        // Attempt 03 of adcf448 held at load1 1.19, then the accuracy phase ran,
        // and the pre-energy recheck saw 3.76–4.96. load1 is a one-minute average,
        // so that order is the harness refusing its own work. The threshold stays 2.
        assert_eq!(phase_order(true), ["energy", "accuracy"]);
        assert_eq!(phase_order(false), ["accuracy", "energy"]);
        assert!(phase_order(true)[0] == "energy");
        assert_eq!(QUIET_LOAD, 2.0);
    }

    #[test]
    fn tegrastats_vdd_in_is_one_rail_and_other_rails_are_not_summed() {
        let line = "RAM 1543/7772MB CPU [1%@102] VDD_IN 1864/1864 VDD_CPU_GPU_CV 341/341 VDD_SOC 717/717";
        assert!((parse_tegrastats_vdd_in_watts(line).unwrap() - 1.864).abs() < 1e-12);
        let mw = "VDD_IN 2500mW/2500mW VDD_CPU_GPU_CV 100mW/100mW";
        assert!((parse_tegrastats_vdd_in_watts(mw).unwrap() - 2.5).abs() < 1e-12);
        assert!(parse_tegrastats_vdd_in_watts("VDD_CPU_GPU_CV 341/341 VDD_SOC 717/717").is_none());
        assert!(parse_tegrastats_vdd_in_watts("no power here").is_none());
    }

    #[test]
    fn a_negative_or_drifting_bracket_is_not_attributable_and_pairs_keep_their_sign() {
        let neg = trial_joules(4.0, 4.0, 0.01, 2.0, "rapl", "package", GRADE_CONTENDED, MIN_COUNTER_UPDATES, true);
        assert!(neg.incremental.unwrap() < 0.0);
        assert!(!neg.attributable);
        assert!(joules_per_ess(&neg, 100.0).is_none());
        let drifted = trial_joules(20.0, 4.0, 0.01, 2.0, "rapl", "package", GRADE_CONTENDED, MIN_COUNTER_UPDATES, false);
        assert!(!drifted.attributable);
        let (mean, sigma, drift, ok) = combine_bracket(10.0, 0.2, 10.1, 0.2);
        assert!((mean - 10.05).abs() < 1e-12);
        assert!(ok, "0.1 W drift against sigma 0.2 is inside 3 sigma, drift {drift} sigma {sigma}");
        let (_, _, big, ok) = combine_bracket(10.0, 0.1, 12.0, 0.1);
        assert!(!ok && (big - 2.0).abs() < 1e-12);
        let delta = paired_delta(1.5, 4.0);
        assert!((delta + 2.5).abs() < 1e-12);
        assert!(paired_attributable(delta, 0.05, 2.0, 2.0, true, MIN_COUNTER_UPDATES, MIN_COUNTER_UPDATES));
        assert!(!paired_attributable(0.01, 0.05, 2.0, 2.0, true, MIN_COUNTER_UPDATES, MIN_COUNTER_UPDATES));
        assert!(!paired_attributable(delta, 0.05, 2.0, 2.0, false, MIN_COUNTER_UPDATES, MIN_COUNTER_UPDATES));
        assert_eq!(pair_order(0), ["ferrotherm", "thrml"]);
        assert_eq!(pair_order(1), ["thrml", "ferrotherm"]);
        assert!(!scopes_may_share_a_table("package", "core"));
        assert!(scopes_may_share_a_table("core", "core"));
    }

    #[test]
    fn cpu_pin_lists_parse_and_other_cores_are_the_ones_outside_the_set() {
        assert_eq!(parse_cpu_list("0-2,8").unwrap(), vec![0, 1, 2, 8]);
        assert!(parse_cpu_list("").is_err());
        assert!(parse_cpu_list("3-1").is_err());
        let before = "cpu  0 0 0 0 0 0 0 0\ncpu0 10 0 0 80 0 0 0 10\ncpu1 10 0 0 90 0 0 0 0\ncpu2 0 0 0 100 0 0 0 0\n";
        let after = "cpu  0 0 0 0 0 0 0 0\ncpu0 30 0 0 80 0 0 0 10\ncpu1 60 0 0 140 0 0 0 0\ncpu2 0 0 0 200 0 0 0 0\n";
        let busy = other_core_busy(before, after, &[0]);
        assert_eq!(busy.len(), 2);
        assert_eq!(busy[0].0, 1);
        assert!((busy[0].1 - 0.5).abs() < 1e-12, "cpu1 busy {}", busy[0].1);
        assert_eq!(busy[1].0, 2);
        assert!(busy[1].1.abs() < 1e-12);
    }
}
