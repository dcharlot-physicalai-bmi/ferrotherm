//! Measured joules, or an explicit absence.
//!
//! Instruments, in the order a host actually has them:
//!
//! - macOS: rootless `macmon` (`IOReport`), via `ferrotherm_meter`. No other macOS path.
//! - Linux CPU: RAPL `energy_uj` under `/sys/class/powercap` when that file is readable.
//!   AMD package and core are model- or estimate-based `SoC` energy, not wall power.
//!   Intel package and core are a different counter and are not given that label.
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

use ferrotherm::host::{QUIET_LOAD, load_average};
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
pub fn split_incremental(
    gross: f64,
    idle_watts: f64,
    idle_sigma: f64,
    seconds: f64,
) -> (f64, bool) {
    let incremental = gross - idle_watts * seconds;
    let band = NOISE_SIGMA_K * idle_sigma * seconds;
    let attributable =
        incremental > band && seconds > 0.0 && idle_sigma.is_finite() && incremental.is_finite();
    (incremental, attributable)
}

/// Mean and sigma of an idle bracket, and whether the two ends drifted too far apart.
///
/// Sigma is the root-mean-square of the two window sigmas. The drift test uses that sigma,
/// so a quiet pair of readings can fail a large jump and a noisy pair is judged against its
/// own wander.
pub fn combine_bracket(
    before_w: f64,
    before_s: f64,
    after_w: f64,
    after_s: f64,
) -> (f64, f64, f64, bool) {
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
    if pair.is_multiple_of(2) {
        ["ferrotherm", "thrml"]
    } else {
        ["thrml", "ferrotherm"]
    }
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
        let Some((cpu, busy, total)) = parse_cpu_n(line) else {
            continue;
        };
        if pinned.contains(&cpu) {
            continue;
        }
        let Some((b0, t0)) = prior.get(&cpu) else {
            continue;
        };
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
        SampledWindow {
            readings,
            seconds,
            cadence_s: median_positive(&dts),
            value,
        }
    })
}

fn median_positive(xs: &[f64]) -> f64 {
    let mut v: Vec<f64> = xs
        .iter()
        .copied()
        .filter(|x| x.is_finite() && *x > 0.0)
        .collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n.is_multiple_of(2) {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    } else {
        v[n / 2]
    }
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
    a == b
        && (matches!(
            a,
            "package"
                | "core"
                | "soc-telemetry"
                | "whole-system"
                | "psys"
                | "board"
                | "vin"
                | "gpu"
        ) || a.starts_with("subrail:"))
}

/// What the scope does not include. This is a statement, not a joule, and it is not folded
/// into the combined uncertainty.
pub fn scope_limit(scope: &str) -> &'static str {
    match scope {
        "soc-telemetry" => {
            "macmon sys_power is SoC telemetry (CPU, integrated GPU, ANE). It is not wall power. Display, fans, and the rest of the board are outside it. Package and core are different scopes and are not added here."
        }
        "package" => {
            "RAPL package energy, not wall power. The core domain is its own scope and is not added. On AMD this is a model- or estimate-based SoC energy counter and the interface has no DRAM domain. On Intel, DRAM is a separate domain when the chip exposes it and is not part of this package figure. The two vendors are not the same counter."
        }
        "core" => {
            "RAPL core domain only. It is not added to package energy and it is not wall power. On AMD it is a model- or estimate-based SoC energy counter and there is no DRAM domain beside it. On Intel it is not the AMD model basis."
        }
        "psys" => {
            "RAPL psys, kept only when it contains the package. It is not macmon and it is not a wall-plug measurement."
        }
        "board" => {
            "One board rail (INA3221 total or tegrastats VDD_IN). Other rails are not summed."
        }
        "vin" => {
            "Module-input boundary. L4T R39 calls INA238 VIN the overall system, module plus carrier, not the wall. INA3221 sub-rails are not added to it."
        }
        "gpu" => {
            "NVML total energy, summed across devices. It is not package energy and not SoC telemetry. On Thor this counter is NOT_SUPPORTED and is not substituted with a sub-rail sum."
        }
        "whole-system" => {
            "Legacy label. On a macmon receipt this token meant sys_power, which is SoC telemetry, not wall power."
        }
        _ if scope.starts_with("subrail:") => {
            "One INA3221 channel. It is not a total, and it is not added to VIN or to the other channels."
        }
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
    if window_s <= 0.0
        || !cadence_s.is_finite()
        || !gross_window.is_finite()
        || !cadence_s.is_sign_positive()
    {
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
    for term in [
        Some(student_t_half),
        quantization,
        cadence_edge,
        Some(bracket),
    ]
    .into_iter()
    .flatten()
    {
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
    let busy = if busy_readable {
        other_core_busy(&before, &after, pinned)
    } else {
        Vec::new()
    };
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
    if quiet_gate_held {
        ["energy", "accuracy"]
    } else {
        ["accuracy", "energy"]
    }
}

/// How often the energy phase reads `load1`. One sample a second, for the whole phase.
pub const ENERGY_WATCH_INTERVAL: Duration = Duration::from_secs(1);

/// One `load1` sample and the time it was taken.
#[derive(Clone, Debug, PartialEq)]
pub struct LoadReading {
    /// Seconds since the Unix epoch, with millisecond fraction. The clock is the host's.
    pub at_unix_s: f64,
    pub load1: f64,
}

/// What an energy-phase load series decides.
///
/// Quiet requires every sample to be finite and at or under [`QUIET_LOAD`]. The first sample
/// that fails aborts the phase. Samples after that are not a recovery, and an empty series is
/// not quiet.
#[derive(Clone, Debug, PartialEq)]
pub struct EnergyWatchVerdict {
    pub evidence_grade: &'static str,
    pub abort: Option<LoadReading>,
    /// Samples up to and including the aborting one.
    pub samples: Vec<LoadReading>,
}

/// Grade `samples` in order. `None` is an unreadable load and aborts.
pub fn grade_energy_watch(samples: &[Option<f64>]) -> EnergyWatchVerdict {
    let mut kept = Vec::new();
    for (i, sample) in samples.iter().enumerate() {
        let at = i as f64;
        match sample {
            Some(v) if v.is_finite() && *v <= QUIET_LOAD => {
                kept.push(LoadReading {
                    at_unix_s: at,
                    load1: *v,
                });
            }
            Some(v) => {
                let reading = LoadReading {
                    at_unix_s: at,
                    load1: if v.is_finite() { *v } else { f64::NAN },
                };
                kept.push(reading.clone());
                return EnergyWatchVerdict {
                    evidence_grade: "load_gated",
                    abort: Some(reading),
                    samples: kept,
                };
            }
            None => {
                let reading = LoadReading {
                    at_unix_s: at,
                    load1: f64::NAN,
                };
                return EnergyWatchVerdict {
                    evidence_grade: "load_gated",
                    abort: Some(reading),
                    samples: kept,
                };
            }
        }
    }
    if kept.is_empty() {
        EnergyWatchVerdict {
            evidence_grade: "load_gated",
            abort: None,
            samples: kept,
        }
    } else {
        EnergyWatchVerdict {
            evidence_grade: GRADE_QUIET,
            abort: None,
            samples: kept,
        }
    }
}

/// The bundle grade is the energy-phase grade.
///
/// A quiet preflight does not make the bundle quiet. [`GRADE_QUIET`] is returned only when the
/// energy phase itself is quiet. [`GRADE_CONTENDED`] stays contended. Anything else is
/// `load_gated`.
pub fn bundle_grade(energy_grade: &str) -> &'static str {
    if energy_grade == GRADE_QUIET {
        GRADE_QUIET
    } else if energy_grade == GRADE_CONTENDED {
        GRADE_CONTENDED
    } else {
        "load_gated"
    }
}

/// Consecutive 1 Hz samples at or under [`QUIET_LOAD`] that an idle bracket waits for.
pub const SETTLE_STREAK: usize = 60;

/// Most 1 Hz samples a settle may take. Past this the bracket is refused.
pub const SETTLE_BOUND: usize = 600;

/// Result of waiting for [`SETTLE_STREAK`] quiet load samples.
#[derive(Clone, Debug, PartialEq)]
pub struct SettleOutcome {
    pub settled: bool,
    pub samples: Vec<f64>,
    pub reason: &'static str,
}

/// Walk `samples` in order, at most [`SETTLE_BOUND`] of them unless `bound` is passed.
///
/// A finite sample at or under [`QUIET_LOAD`] extends the streak. Anything else resets it.
/// The wait succeeds when the streak reaches `streak` samples.
pub fn settle_outcome(samples: &[Option<f64>], streak: usize, bound: usize) -> SettleOutcome {
    let mut kept = Vec::new();
    let mut run = 0usize;
    if streak == 0 {
        return SettleOutcome {
            settled: false,
            samples: kept,
            reason: "the settle streak is zero",
        };
    }
    for sample in samples.iter().take(bound) {
        match sample {
            Some(v) if v.is_finite() && *v <= QUIET_LOAD => {
                kept.push(*v);
                run += 1;
                if run >= streak {
                    return SettleOutcome {
                        settled: true,
                        samples: kept,
                        reason: "load1 stayed at or under 2 for the required consecutive samples",
                    };
                }
            }
            Some(v) => {
                kept.push(if v.is_finite() { *v } else { f64::NAN });
                run = 0;
            }
            None => {
                kept.push(f64::NAN);
                run = 0;
            }
        }
    }
    SettleOutcome {
        settled: false,
        samples: kept,
        reason: "load1 did not stay at or under 2 for 60 consecutive 1 Hz samples within the bound",
    }
}

/// `Some(load)` when a bracket sample is missing or above [`QUIET_LOAD`].
///
/// An empty series has not been watched, so it is a refusal. The returned load is the
/// first offending sample.
pub fn bracket_load_offender(samples: &[Option<f64>]) -> Option<f64> {
    if samples.is_empty() {
        return Some(f64::NAN);
    }
    for sample in samples {
        match sample {
            Some(v) if v.is_finite() && *v <= QUIET_LOAD => {}
            Some(v) => return Some(if v.is_finite() { *v } else { f64::NAN }),
            None => return Some(f64::NAN),
        }
    }
    None
}

/// Thresholds stamped on every energy receipt. The trial-window rule uses these.
///
/// `load1` is not one of them. It counts runnable threads, so the multithreaded
/// workload under test trips it. Brackets use `load1`. Trial windows do not.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrialWindowRule {
    /// When false, samples are recorded and no trial is refused for them.
    pub enforce: bool,
    /// Linux: mean busy fraction of cores outside `--cpus`.
    pub other_core_mean_max: f64,
    /// Linux: no 1 Hz sample of that fraction above this.
    pub other_core_sample_max: f64,
    /// macOS: mean summed non-harness CPU, as a fraction of one core.
    pub foreign_cpu_mean_max: f64,
    /// macOS: no 1 Hz sample of that sum above this.
    pub foreign_cpu_sample_max: f64,
}

impl TrialWindowRule {
    pub fn defaults() -> TrialWindowRule {
        TrialWindowRule {
            enforce: true,
            other_core_mean_max: 0.05,
            other_core_sample_max: 0.15,
            foreign_cpu_mean_max: 0.15,
            foreign_cpu_sample_max: 0.50,
        }
    }
}

/// Why a 1 Hz fraction series is refused, or `None` when it passes.
///
/// Empty input is a refusal: a window that recorded nothing did not clear the rule.
pub fn fraction_refusal(samples: &[f64], mean_max: f64, sample_max: f64) -> Option<&'static str> {
    if samples.is_empty() || samples.iter().any(|s| !s.is_finite()) {
        return Some("the trial window recorded no finite sample");
    }
    if samples.iter().any(|s| *s > sample_max) {
        return Some("a sample exceeded the maximum");
    }
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    if mean > mean_max {
        return Some("the mean exceeded the maximum");
    }
    None
}

/// Linux quiet energy needs a pin. Unpinned, every core is a core the sampler may run on,
/// and the outside-core fraction is undefined.
pub fn linux_quiet_needs_pin(os: &str, pinned: usize) -> bool {
    os == "linux" && pinned == 0
}

/// macOS quiet energy requires the idle bracket's drift test. `drift_ok` is the existing
/// 3-sigma comparison. A false result refuses the quiet grade. It does not change joules.
pub fn macos_drift_refuses_quiet(os: &str, drift_ok: bool) -> bool {
    os == "macos" && !drift_ok
}

/// Mean busy fraction of the cores outside `pinned`, from two `/proc/stat` snapshots.
///
/// No outside core means the fraction is 0. A snapshot pair with no parseable ticks is `None`.
pub fn other_core_fraction(before: &str, after: &str, pinned: &[usize]) -> Option<f64> {
    if !before.contains("cpu") || !after.contains("cpu") {
        return None;
    }
    let cores = other_core_busy(before, after, pinned);
    if cores.is_empty() {
        return Some(0.0);
    }
    Some(cores.iter().map(|(_, fraction)| fraction).sum::<f64>() / cores.len() as f64)
}

/// Exclusive host lock. `flock(2)` `LOCK_EX` on Linux and on macOS.
///
/// The constants are the same on both: `LOCK_EX` is 2, `LOCK_NB` is 4, `LOCK_UN` is 8.
pub const HOST_LOCK_DEFAULT: &str = "/tmp/bmi-energy.lock";

/// Default bound, in seconds, for waiting on a busy host lock.
pub const HOST_LOCK_WAIT_S: f64 = 3600.0;

/// Refusal text when the lock is not acquired. The run does not continue unlocked.
pub const HOST_LOCK_REFUSAL: &str = "host lock not held";

const LOCK_EX: i32 = 2;
const LOCK_NB: i32 = 4;
const LOCK_UN: i32 = 8;

/// `seconds` since the Unix epoch as `YYYY-MM-DDThh:mm:ss.mmmZ`.
pub fn format_unix_utc(seconds: f64) -> String {
    if !seconds.is_finite() {
        return "invalid".into();
    }
    let whole = seconds.floor() as i64;
    let millis = ((seconds - whole as f64) * 1000.0)
        .round()
        .clamp(0.0, 999.0) as u32;
    let days = whole.div_euclid(86400);
    let tod = whole.rem_euclid(86400) as u32;
    let hour = tod / 3600;
    let min = (tod % 3600) / 60;
    let sec = tod % 60;
    let (year, month, day) = civil_from_days(days);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{min:02}:{sec:02}.{millis:03}Z")
}

fn civil_from_days(days: i64) -> (i32, u32, u32) {
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };
    (year as i32, month as u32, day as u32)
}

/// A held `flock(2)` `LOCK_EX`. Dropping it unlocks.
pub struct HostLock {
    file: std::fs::File,
    pub path: std::path::PathBuf,
    pub acquired_unix_s: f64,
    pub wait_s: f64,
    pub holder_pid: u32,
}

impl HostLock {
    /// Block until `LOCK_EX` is held, or return [`HOST_LOCK_REFUSAL`] when `wait_s` elapses.
    ///
    /// The wait is polled with `LOCK_NB` once a second and logged. A failed acquire does not
    /// leave the caller holding the lock.
    pub fn acquire(path: &std::path::Path, wait_s: f64) -> Result<HostLock, String> {
        #[cfg(not(unix))]
        {
            let _ = (path, wait_s);
            return Err(HOST_LOCK_REFUSAL.into());
        }
        #[cfg(unix)]
        {
            acquire_unix(path, wait_s)
        }
    }

    pub fn acquired_utc(&self) -> String {
        format_unix_utc(self.acquired_unix_s)
    }

    /// Unlock, then return the UTC time the lock was released.
    pub fn release(self) -> String {
        drop(self);
        format_unix_utc(unix_now_s())
    }
}

impl Drop for HostLock {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            let _ = unlock_fd(std::os::unix::io::AsRawFd::as_raw_fd(&self.file));
        }
    }
}

#[cfg(unix)]
fn acquire_unix(path: &std::path::Path, wait_s: f64) -> Result<HostLock, String> {
    use std::io::{Seek, Write};
    use std::os::unix::io::AsRawFd;
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        let _ = std::fs::create_dir_all(parent);
    }
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .map_err(|e| {
            eprintln!("host lock {path:?} was not opened: {e}");
            HOST_LOCK_REFUSAL.to_string()
        })?;
    let fd = file.as_raw_fd();
    let started = std::time::Instant::now();
    let wait = if wait_s.is_finite() && wait_s > 0.0 {
        wait_s
    } else {
        0.0
    };
    loop {
        if flock_nb(fd) {
            let pid = std::process::id();
            let mut owned = file;
            let _ = owned.seek(std::io::SeekFrom::Start(0));
            let _ = owned.set_len(0);
            let _ = writeln!(owned, "{pid}");
            let _ = owned.flush();
            return Ok(HostLock {
                file: owned,
                path: path.to_path_buf(),
                acquired_unix_s: unix_now_s(),
                wait_s: started.elapsed().as_secs_f64(),
                holder_pid: pid,
            });
        }
        let holder = std::fs::read_to_string(path)
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok());
        let waited = started.elapsed().as_secs_f64();
        let holder_txt = match holder {
            Some(pid) => pid.to_string(),
            None => "unknown".to_string(),
        };
        eprintln!(
            "host lock busy: {} holder_pid={holder_txt} waited={waited:.1}s",
            path.display()
        );
        if waited >= wait {
            return Err(HOST_LOCK_REFUSAL.into());
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}

#[cfg(unix)]
fn flock_nb(fd: i32) -> bool {
    unsafe { flock(fd, LOCK_EX | LOCK_NB) == 0 }
}

#[cfg(unix)]
fn unlock_fd(fd: i32) -> bool {
    unsafe { flock(fd, LOCK_UN) == 0 }
}

#[cfg(unix)]
unsafe extern "C" {
    fn flock(fd: i32, operation: i32) -> i32;
}

/// `ps` cumulative CPU time. Accepts `m:ss`, `m:ss.ff`, `h:mm:ss`, and `d-hh:mm:ss`.
pub fn parse_cpu_clock(text: &str) -> Option<f64> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    let (days, rest) = match text.split_once('-') {
        Some((d, rest)) => (d.parse::<f64>().ok()?, rest),
        None => (0.0, text),
    };
    let mut parts = rest.split(':').collect::<Vec<_>>();
    if parts.len() == 1 {
        return None;
    }
    let sec: f64 = parts.pop()?.parse().ok()?;
    let min: f64 = parts.pop()?.parse().ok()?;
    let hour: f64 = if let Some(h) = parts.pop() {
        h.parse().ok()?
    } else {
        0.0
    };
    if !parts.is_empty() {
        return None;
    }
    Some(days * 86400.0 + hour * 3600.0 + min * 60.0 + sec)
}

/// One `ps` row: pid, parent pid, cumulative CPU seconds.
pub fn parse_ps_cpu_row(line: &str) -> Option<(u32, u32, f64)> {
    let mut it = line.split_whitespace();
    let pid: u32 = it.next()?.parse().ok()?;
    let ppid: u32 = it.next()?.parse().ok()?;
    let time = parse_cpu_clock(it.next()?)?;
    Some((pid, ppid, time))
}

/// Sum of CPU-seconds gained by processes that are not `self_pid` and not its descendants,
/// divided by `wall_s`. That is a fraction of one core. It is a refusal input, not an energy term.
pub fn foreign_cpu_fraction(
    before: &[(u32, u32, f64)],
    after: &[(u32, u32, f64)],
    self_pid: u32,
    wall_s: f64,
) -> Option<f64> {
    if !(wall_s > 0.0) {
        return None;
    }
    let mut parent = std::collections::BTreeMap::new();
    for (pid, ppid, _) in before.iter().chain(after.iter()) {
        parent.insert(*pid, *ppid);
    }
    let harness = |pid: u32| -> bool {
        let mut cur = pid;
        for _ in 0..32 {
            if cur == self_pid {
                return true;
            }
            match parent.get(&cur).copied() {
                Some(next) if next != cur && next != 0 => cur = next,
                _ => return false,
            }
        }
        false
    };
    let mut prior = std::collections::BTreeMap::new();
    for (pid, _, cpu) in before {
        prior.insert(*pid, *cpu);
    }
    let mut sum = 0.0;
    for (pid, _, cpu) in after {
        if harness(*pid) {
            continue;
        }
        if let Some(was) = prior.get(pid) {
            sum += (cpu - was).max(0.0);
        }
    }
    Some(sum / wall_s)
}

fn unix_now_s() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(f64::NAN, |d| d.as_secs_f64())
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
                return QuietPoll {
                    outcome: "quiet",
                    samples,
                    accepted: Some(*l),
                    threshold: QUIET_LOAD,
                };
            }
            Some(l) => samples.push(*l),
            None => samples.push(f64::NAN),
        }
    }
    QuietPoll {
        outcome: "load_gated",
        samples,
        accepted: None,
        threshold: QUIET_LOAD,
    }
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
    let v: Vec<f64> = watts
        .iter()
        .copied()
        .filter(|w| w.is_finite() && *w >= 0.0)
        .collect();
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
    if now >= prev {
        now - prev
    } else {
        (max_j - prev) + now
    }
}

/// Watts from successive counter windows `(delta_joules, seconds)`.
pub fn watts_from_windows(windows: &[(f64, f64)]) -> Option<(f64, f64)> {
    let watts: Vec<f64> = windows
        .iter()
        .filter(|(j, s)| *s > 0.0 && j.is_finite())
        .map(|(j, s)| j / s)
        .collect();
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
    if saw {
        Ok(config)
    } else {
        Err(format!("no perf event field in {text:?}"))
    }
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

/// RAPL on `framework-128gb-0000` is the instrument that host reads.
///
/// That rank is not a wall-validated absolute joule. The host is AMD, and
/// [`rapl_basis`] says what the counter is. macmon is secondary. A different
/// Framework hostname is not this machine.
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

/// `AuthenticAMD`, `GenuineIntel`, or `unknown` from `/proc/cpuinfo`.
pub fn cpu_vendor() -> &'static str {
    cpu_vendor_from(&std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default())
}

/// Vendor id in a `cpuinfo` blob. `unknown` when the line is absent.
pub fn cpu_vendor_from(cpuinfo: &str) -> &'static str {
    for line in cpuinfo.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        if key.trim() == "vendor_id" {
            return match value.trim() {
                "AuthenticAMD" => "amd",
                "GenuineIntel" => "intel",
                _ => "unknown",
            };
        }
    }
    "unknown"
}

/// What a RAPL package or core counter is, for the vendor that produced it.
///
/// AMD and Intel are not given the same sentence. Absolute joules per effective
/// sample from the AMD counter are not a wall figure.
pub fn rapl_basis(vendor: &str) -> &'static str {
    match vendor {
        "amd" => {
            "AMD package and core RAPL are model- or estimate-based SoC energy counters, not wall power, and this interface has no DRAM domain. Schöne et al., IEEE CLUSTER 2021 (arXiv:2108.00808), on a Zen 2 EPYC 7502: the energy data is modeled, not measured. Zen 5 / Strix Halo accuracy (framework-128gb-0000, Ryzen AI Max+ 395) has not been validated by any primary measurement study. The kernel amd_energy hwmon text calls the inputs Measured and does not say how. This note does not claim that AMD documentation says the counters are modeled. Absolute J/ESS from these counters needs a wall-meter cross-check before it can be claimed. A same-host ratio may stand when this model basis is stated with it."
        }
        "intel" => {
            "Intel RAPL package and core are that processor's RAPL energy counters for those domains. They are not wall power. DRAM, when the chip exposes it, is a separate domain and is not part of package or core. This is not the AMD model or estimate basis."
        }
        _ => {
            "RAPL vendor was not read from cpuinfo, so this note does not call the counter AMD or Intel. It is not wall power."
        }
    }
}

/// `macmon --version`, or why the binary did not answer. Stamped on every receipt.
pub fn macmon_version() -> String {
    match std::process::Command::new("macmon").arg("--version").output() {
        Ok(out) if out.status.success() => {
            let text = String::from_utf8_lossy(&out.stdout);
            let text = text.trim();
            if text.is_empty() {
                let err = String::from_utf8_lossy(&out.stderr);
                let err = err.trim();
                if err.is_empty() {
                    "macmon --version produced no text".into()
                } else {
                    err.to_string()
                }
            } else {
                text.lines().next().unwrap_or(text).to_string()
            }
        }
        Ok(out) => {
            let err = String::from_utf8_lossy(&out.stderr);
            format!("macmon --version exited {}: {}", out.status, err.trim())
        }
        Err(e) => format!("macmon not run: {e}"),
    }
}

/// Other-process CPU time, in seconds, between two snapshots.
///
/// `busy` is every non-idle tick on the machine. `mine` is this process. The difference is
/// everyone else. A wrap or a clock that went backwards contributes zero, not a negative.
pub fn other_cpu_seconds(
    busy_before: u64,
    mine_before: u64,
    busy_after: u64,
    mine_after: u64,
    hz: f64,
) -> f64 {
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
        "attribution refused: the idle bracket drifted, or the quiet gate failed on a bracket"
            .to_string()
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
    /// 1 Hz `load1` through the trial window, comma-separated. Evidence only.
    pub window_load1: String,
    /// 1 Hz trial-window signal: outside-core busy fraction, or non-harness CPU fraction.
    pub window_signal: String,
    /// `ps` rows for processes that are not this harness. Attribution, not a joule term.
    pub ps_attribution: String,
    /// Empty when the trial window passed. Otherwise the refusal text.
    pub trial_refusal: String,
    /// 1 Hz `load1` samples of the settle that preceded this pair's bracket.
    pub settle_load1: String,
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
            window_load1: String::new(),
            window_signal: String::new(),
            ps_attribution: String::new(),
            trial_refusal: String::new(),
            settle_load1: String::new(),
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
    let mean = |v: &[f64]| {
        if v.is_empty() {
            None
        } else {
            Some(v.iter().sum::<f64>() / v.len() as f64)
        }
    };
    combine_uncertainty(
        student,
        mean(&quant),
        mean(&edge),
        mean(&bracket).unwrap_or(f64::NAN),
    )
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
        let key = (
            t.workload,
            t.side,
            t.joules.instrument.as_str(),
            t.joules.scope.as_str(),
            t.grade,
        );
        if !keys.contains(&key) {
            keys.push(key);
        }
    }
    keys.into_iter()
        .filter_map(|key| {
            let group: Vec<EnergyTrial> = trials
                .iter()
                .filter(|t| {
                    (
                        t.workload,
                        t.side,
                        t.joules.instrument.as_str(),
                        t.joules.scope.as_str(),
                        t.grade,
                    ) == key
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
    joules
        .incremental
        .filter(|j| j.is_finite())
        .map(|j| j / ess)
}

pub struct Probe {
    pub load1: Option<f64>,
    pub quiet: bool,
    /// macmon or INA3221. A RAPL meter is not kept here: RAPL is read as a counter below.
    pub wall: Option<Meter>,
    pub wall_note: String,
    pub rapl: Option<RaplCounter>,
    pub rapl_note: String,
    /// The `core` subdomain. Not added to the package counter.
    pub rapl_core: Option<RaplCounter>,
    pub rapl_core_note: String,
    /// Vendor label for package and core. AMD and Intel do not share this sentence.
    pub rapl_basis: String,
    /// Output of `macmon --version`, or why it did not run.
    pub macmon_version: String,
    pub perf: Option<PerfPkg>,
    pub perf_note: String,
    pub nvml: Option<Nvml>,
    pub nvml_note: String,
    /// INA238 `VIN` when that hwmon node is present. Not wall power, and not a sum of sub-rails.
    pub ina238: Option<HwmonSeries>,
    pub ina238_note: String,
    /// INA3221 channels that are not a total. Each is its own scope.
    pub subrails: Vec<HwmonSeries>,
    /// Jetson `tegrastats` `VIN` or `VDD_IN`, only when INA238 did not open. Not a sum of rails.
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
        let macmon_version = macmon_version();
        let wall_note = match &wall {
            Some(m) if m.backend() == "macmon" => format!(
                "backend=macmon version={macmon_version} machine={} scope={}",
                m.machine(),
                m.scope().label()
            ),
            Some(m) => format!(
                "backend={} machine={} scope={}",
                m.backend(),
                m.machine(),
                m.scope().label()
            ),
            None => "no macmon and no INA3221 rail".into(),
        };
        let basis = rapl_basis(cpu_vendor()).to_string();
        let (rapl, rapl_note) = match RaplCounter::open() {
            Ok(c) => {
                let note = format!("{} ({}). {basis}", c.name, c.kind);
                (Some(c), note)
            }
            Err(e) => (None, e),
        };
        let (rapl_core, rapl_core_note) = match rapl
            .as_ref()
            .and_then(|c| RaplCounter::open_core(c.counter_path()))
        {
            Some(c) => (
                Some(c),
                format!("core subdomain, reported separately from package. {basis}"),
            ),
            None => (None, format!("no readable core subdomain. {basis}")),
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
        let (ina238, ina238_note) = match HwmonSeries::open_ina238() {
            Ok(series) => {
                let note = series.note.clone();
                (Some(series), note)
            }
            Err(e) => (None, format!("backend not chosen: ina238. reason: {e}")),
        };
        let subrails = HwmonSeries::open_subrails();
        let (tegra, tegra_note) = if ina238.is_some() {
            (
                None,
                "backend not chosen: tegrastats. reason: INA238 VIN is the Jetson module-input meter (module plus carrier, not wall power), so tegrastats is not opened.".into(),
            )
        } else if wall.is_some() {
            (
                None,
                "backend not chosen: tegrastats. reason: a power backend (macmon SoC telemetry or an INA3221 total) is already open.".into(),
            )
        } else {
            match Tegra::open() {
                Ok(t) => {
                    let note = format!(
                        "backend chosen: tegrastats. reason: {ina238_note} label {} is Jetson module input (VIN or VDD_IN), not wall power. Other rails are not summed.",
                        t.label
                    );
                    (Some(t), note)
                }
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
            rapl_basis: basis,
            macmon_version,
            perf,
            perf_note,
            nvml,
            nvml_note,
            ina238,
            ina238_note,
            subrails,
            tegra,
            tegra_note,
            hostname: read_hostname(),
            macos_policy: "macOS power, when that OS is the host, is ferrotherm_meter::Meter::macmon \
                           (rootless IOReport via the macmon binary). This harness has no other macOS path.",
        }
    }
}

/// Module-input sample from one `tegrastats` line.
///
/// `label` is the token that matched: `VIN` on Thor, `VDD_IN` on earlier Jetsons. `VIN_SYS_5V0`
/// is a different token and is not this rail. The number before the slash is instantaneous
/// milliwatts. Sub-rails on the same line are not added.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TegraInput {
    /// `VIN` or `VDD_IN`.
    pub label: &'static str,
    /// Instantaneous draw, in watts.
    pub watts: f64,
}

/// `VIN` if the line has it, otherwise `VDD_IN`.
///
/// Thor prints module input as `VIN`. A line that only names `VDD_GPU`, `VDD_CPU_SOC_MSS`, or
/// `VIN_SYS_5V0` is `None`.
pub fn parse_tegrastats_module_input(line: &str) -> Option<TegraInput> {
    let mut vin = None;
    let mut vdd = None;
    let mut tokens = line.split_whitespace();
    while let Some(tok) = tokens.next() {
        let label = match tok {
            "VIN" => "VIN",
            "VDD_IN" => "VDD_IN",
            _ => continue,
        };
        let Some(raw) = tokens.next() else { continue };
        let head = raw.split('/').next().unwrap_or(raw);
        let digits: String = head
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.')
            .collect();
        let Ok(mw) = digits.parse::<f64>() else {
            continue;
        };
        if mw.is_finite() && mw >= 0.0 {
            let sample = TegraInput {
                label,
                watts: mw / 1000.0,
            };
            if label == "VIN" {
                vin = Some(sample);
            } else {
                vdd = Some(sample);
            }
        }
    }
    vin.or(vdd)
}

/// A rootless `tegrastats` reader. The child is stopped when this value is dropped.
pub struct Tegra {
    child: std::process::Child,
    samples: std::sync::Arc<std::sync::Mutex<Vec<(std::time::Instant, f64)>>>,
    /// `VIN` or `VDD_IN`, whichever the first sample matched.
    pub label: String,
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
        let label = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let sink = std::sync::Arc::clone(&samples);
        let sink_label = std::sync::Arc::clone(&label);
        std::thread::spawn(move || {
            let mut reader = std::io::BufReader::new(stdout);
            let mut line = String::new();
            loop {
                line.clear();
                let n = std::io::BufRead::read_line(&mut reader, &mut line).unwrap_or(0);
                if n == 0 {
                    break;
                }
                if let Some(sample) = parse_tegrastats_module_input(&line) {
                    let mut name = sink_label.lock().unwrap();
                    if name.is_empty() {
                        *name = sample.label.to_string();
                    }
                    drop(name);
                    sink.lock()
                        .unwrap()
                        .push((std::time::Instant::now(), sample.watts));
                }
            }
        });
        let start = std::time::Instant::now();
        while start.elapsed() < Duration::from_secs(2) {
            if !samples.lock().unwrap().is_empty() {
                let label = label.lock().unwrap().clone();
                return Ok(Tegra {
                    child,
                    samples,
                    label,
                });
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = child.kill();
        Err("tegrastats produced no VIN or VDD_IN sample. Other rails were not summed.".into())
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

/// An `ina238` hwmon node, identified by its `name` file and i2c address.
///
/// `power1_input` is microwatts. `energy1_input`, when present, is microjoules and is preferred
/// because it is a counter. L4T R39's Jetson Thor guide says the INA238 at i2c `2-0044` is `VIN`:
/// overall system power, module plus carrier. That is not wall power. A different address or name
/// is read and marked unverified.
#[derive(Clone, Debug, PartialEq)]
pub struct Ina238Found {
    pub name: String,
    pub i2c: String,
    /// `power1_input` or `energy1_input`.
    pub source: &'static str,
    pub path: std::path::PathBuf,
    /// True only for `ina238` at i2c `2-0044`, the rail L4T R39 documents.
    pub verified: bool,
    /// Cumulative microjoules when `source` is `energy1_input`. Watts samples otherwise.
    pub counter: bool,
    pub note: String,
}

fn i2c_addr_of(start: &std::path::Path) -> String {
    if let Some(addr) = i2c_addr_along(start) {
        return addr;
    }
    // `/sys/class/hwmon/hwmonN` is a symlink. The i2c address is on the target, not the link path.
    if let Ok(canon) = std::fs::canonicalize(start)
        && let Some(addr) = i2c_addr_along(&canon)
    {
        return addr;
    }
    "unknown".into()
}

fn i2c_addr_along(start: &std::path::Path) -> Option<String> {
    for ancestor in start.ancestors() {
        if let Some(name) = ancestor.file_name().and_then(|s| s.to_str())
            && is_i2c_node(name)
        {
            return Some(name.to_string());
        }
    }
    None
}

fn is_i2c_node(name: &str) -> bool {
    let Some((bus, addr)) = name.split_once('-') else {
        return false;
    };
    bus.chars().all(|c| c.is_ascii_digit())
        && addr.len() == 4
        && addr.chars().all(|c| c.is_ascii_hexdigit())
}

/// Walk `dir`, following directory symlinks.
///
/// `symlink_metadata` does not treat a symlink as a directory, so `/sys/class/hwmon/hwmonN`
/// and `/sys/bus/i2c/devices/2-0044` were skipped and the deep device path was past the
/// depth bound. `metadata` follows the link. Canonical paths are remembered so a link
/// back into an ancestor does not recurse.
fn walk_hwmon_names(
    dir: &std::path::Path,
    out: &mut Vec<std::path::PathBuf>,
    seen: &mut std::collections::HashSet<std::path::PathBuf>,
    depth: usize,
) {
    if depth > 8 {
        return;
    }
    let canon = std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());
    if !seen.insert(canon) {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for ent in rd.flatten() {
        let path = ent.path();
        if path.file_name().and_then(|s| s.to_str()) == Some("name") {
            out.push(path);
        } else if std::fs::metadata(&path).is_ok_and(|m| m.is_dir()) {
            walk_hwmon_names(&path, out, seen, depth + 1);
        }
    }
}

/// Find an `ina238` under `root`. Prefers a verified `2-0044` node when several exist.
///
/// Looks through `/sys/class/hwmon/*` symlinks and `/sys/bus/i2c/devices/*/hwmon/*`, and
/// through a direct `2-0044/hwmon/` tree. The same device reached both ways is one result.
pub fn find_ina238(root: &std::path::Path) -> Result<Ina238Found, String> {
    let mut names = Vec::new();
    let mut seen = std::collections::HashSet::new();
    walk_hwmon_names(&root.join("class/hwmon"), &mut names, &mut seen, 0);
    walk_hwmon_names(&root.join("bus/i2c/devices"), &mut names, &mut seen, 0);
    walk_hwmon_names(root, &mut names, &mut seen, 0);
    let mut found = Vec::new();
    for name_path in names {
        let Ok(raw) = std::fs::read_to_string(&name_path) else {
            continue;
        };
        let name = raw.trim().to_string();
        if !name.eq_ignore_ascii_case("ina238") {
            continue;
        }
        let Some(dir) = name_path.parent() else {
            continue;
        };
        let energy = dir.join("energy1_input");
        let power = dir.join("power1_input");
        let (path, source, counter) = if energy.is_file() {
            (energy, "energy1_input", true)
        } else if power.is_file() {
            (power, "power1_input", false)
        } else {
            continue;
        };
        let i2c = i2c_addr_of(dir);
        let verified = i2c.eq_ignore_ascii_case("2-0044");
        let resolved = std::fs::canonicalize(dir)
            .map_or_else(|_| dir.display().to_string(), |p| p.display().to_string());
        let volts = std::fs::read_to_string(dir.join("in0_input"))
            .ok()
            .and_then(|s| s.trim().parse::<f64>().ok())
            .filter(|mv| mv.is_finite())
            .map(|mv| format!(" in0_input={:.3} V", mv / 1000.0))
            .unwrap_or_default();
        let note = if verified {
            format!(
                "backend chosen: ina238. reason: verified Jetson module input (VIN) at i2c {i2c}, module plus carrier, not wall power. name={name} source={source} path={} resolved={resolved}.{volts} INA3221 sub-rails are not added.",
                dir.display()
            )
        } else {
            format!(
                "unverified ina238 i2c {i2c} name={name} source={source} path={} resolved={resolved}.{volts} Not the L4T R39 node at 2-0044, so this is not established as module-input VIN and is not wall power.",
                dir.display()
            )
        };
        let sensor = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
        if found.iter().any(|item: &Ina238Found| {
            std::fs::canonicalize(&item.path).unwrap_or_else(|_| item.path.clone()) == sensor
        }) {
            continue;
        }
        found.push(Ina238Found {
            name,
            i2c,
            source,
            path,
            verified,
            counter,
            note,
        });
    }
    found.sort_by_key(|item| !item.verified);
    found
        .into_iter()
        .next()
        .ok_or_else(|| {
            "no ina238 hwmon with power1_input or energy1_input after following class/hwmon symlinks and bus/i2c/devices/*/hwmon".into()
        })
}

/// Scope token for one INA3221 channel that is not a total rail.
///
/// `None` for a total label, including the removed voltage sum. The label is kept intact so
/// `VDD_GPU` and `VIN_SYS_5V0` stay different scopes.
pub fn subrail_scope(label: &str) -> Option<String> {
    if ferrotherm_meter::ina3221::is_total_label(label)
        || label.eq_ignore_ascii_case("sum of shunt voltages")
    {
        return None;
    }
    let label = label.trim();
    if label.is_empty() {
        return None;
    }
    Some(format!("subrail:{label}"))
}

/// A hwmon series polled on its own thread. `counter` samples are cumulative joules.
pub struct HwmonSeries {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    samples: std::sync::Arc<std::sync::Mutex<Vec<(std::time::Instant, f64)>>>,
    pub scope: String,
    pub instrument: &'static str,
    pub note: String,
    pub counter: bool,
}

impl HwmonSeries {
    fn spawn_file(found: &Ina238Found) -> HwmonSeries {
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let samples = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let flag = std::sync::Arc::clone(&stop);
        let sink = std::sync::Arc::clone(&samples);
        let path = found.path.clone();
        let scale = 1.0e-6;
        std::thread::spawn(move || {
            while !flag.load(std::sync::atomic::Ordering::Relaxed) {
                if let Ok(text) = std::fs::read_to_string(&path)
                    && let Ok(raw) = text.trim().parse::<f64>()
                    && raw.is_finite()
                    && raw >= 0.0
                {
                    sink.lock()
                        .unwrap()
                        .push((std::time::Instant::now(), raw * scale));
                }
                std::thread::sleep(Duration::from_millis(200));
            }
        });
        HwmonSeries {
            stop,
            samples,
            scope: "vin".into(),
            instrument: "ina238",
            note: found.note.clone(),
            counter: found.counter,
        }
    }

    pub fn open_ina238() -> Result<HwmonSeries, String> {
        let found = find_ina238(std::path::Path::new("/sys"))?;
        Ok(Self::spawn_file(&found))
    }

    /// One background reader per non-total INA3221 channel under `/sys`.
    pub fn open_subrails() -> Vec<HwmonSeries> {
        let mut rails = Vec::new();
        for root in ["/sys/bus/i2c/devices", "/sys/class/hwmon"] {
            if let Ok(found) = ferrotherm_meter::ina3221::Rails::list_at(std::path::Path::new(root))
            {
                for rail in found {
                    if subrail_scope(&rail.label).is_some()
                        && rails
                            .iter()
                            .all(|have: &ferrotherm_meter::ina3221::Rail| have.label != rail.label)
                    {
                        rails.push(rail);
                    }
                }
            }
        }
        rails
            .into_iter()
            .filter_map(|rail| {
                let scope = subrail_scope(&rail.label)?;
                let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                let samples = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
                let flag = std::sync::Arc::clone(&stop);
                let sink = std::sync::Arc::clone(&samples);
                std::thread::spawn(move || {
                    while !flag.load(std::sync::atomic::Ordering::Relaxed) {
                        if let Ok(watts) = ferrotherm_meter::ina3221::Rails::channel_watts(&rail)
                            && watts.is_finite()
                            && watts >= 0.0
                        {
                            sink.lock()
                                .unwrap()
                                .push((std::time::Instant::now(), watts));
                        }
                        std::thread::sleep(Duration::from_millis(200));
                    }
                });
                Some(HwmonSeries {
                    stop,
                    samples,
                    scope,
                    instrument: "ina3221",
                    note: "INA3221 sub-rail. Not added to VIN and not added to the other channels."
                        .into(),
                    counter: false,
                })
            })
            .collect()
    }

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

impl Drop for HwmonSeries {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
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
        let pkg = domains
            .iter()
            .position(|(_, n, _)| n.starts_with("package"));
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
        Ok(RaplCounter {
            path,
            max_uj,
            name,
            kind,
        })
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
        counter_delta_j(
            prev as f64 / 1e6,
            now as f64 / 1e6,
            self.max_uj as f64 / 1e6,
        )
    }

    /// The `core` child of a package domain, if its `energy_uj` is readable.
    ///
    /// The `core` child of the package. The kernel directory may be named `intel-rapl`
    /// on an AMD host; that name is not the vendor label. Joules are not added to package.
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
            return Some(RaplCounter {
                path: p,
                max_uj: max,
                name: "core".into(),
                kind: "core",
            });
        }
        None
    }
}

fn watts_over(dir: &std::path::Path, secs: f64) -> Option<f64> {
    let read = |p: &std::path::Path| -> Option<u64> {
        std::fs::read_to_string(p.join("energy_uj"))
            .ok()?
            .trim()
            .parse()
            .ok()
    };
    let a = read(dir)?;
    let t = std::time::Instant::now();
    std::thread::sleep(Duration::from_secs_f64(secs));
    let b = read(dir)?;
    let max: u64 = std::fs::read_to_string(dir.join("max_energy_range_uj"))
        .ok()?
        .trim()
        .parse()
        .ok()?;
    let d = if b >= a {
        b - a
    } else {
        max.saturating_sub(a).saturating_add(b)
    };
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
    if n == 8 {
        Some(u64::from_ne_bytes(buf) as f64 * scale)
    } else {
        None
    }
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
    let scale =
        std::fs::read_to_string("/sys/bus/event_source/devices/power/events/energy-pkg.scale")
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
        return Err(format!(
            "perf_event_open power/energy-pkg: {err}. {}",
            perf_stat_note()
        ));
    }
    Ok(PerfPkg {
        fd: fd as i32,
        scale,
    })
}

fn perf_stat_note() -> String {
    match std::process::Command::new("perf")
        .args(["stat", "-e", "power/energy-pkg/", "--", "true"])
        .output()
    {
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
    let list = cpus
        .iter()
        .map(std::string::ToString::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let pid = std::process::id().to_string();
    match std::process::Command::new("taskset")
        .args(["-c", &list, "-p", &pid])
        .status()
    {
        Ok(s) if s.success() => Ok(()),
        Ok(s) => Err(format!(
            "sched_setaffinity returned {rc} and taskset exited {s}"
        )),
        Err(e) => Err(format!(
            "sched_setaffinity returned {rc} and taskset was not started: {e}"
        )),
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
    let mine = std::fs::read_to_string("/proc/self/stat")
        .ok()
        .and_then(|s| parse_self_stat_busy(&s))
        .unwrap_or(0);
    CpuSnap {
        busy,
        mine,
        load1: load_average(),
    }
}

pub fn snap_other_cpu(before: &CpuSnap, after: &CpuSnap) -> Option<f64> {
    if before.busy == 0 && after.busy == 0 {
        return None;
    }
    Some(other_cpu_seconds(
        before.busy,
        before.mine,
        after.busy,
        after.mine,
        100.0,
    ))
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
        return Err(
            "libnvidia-ml.so.1 did not load (no NVIDIA Management Library on this machine)".into(),
        );
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
    type GetName = unsafe extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_char, u32) -> i32;
    let init: Init = unsafe { std::mem::transmute(sym("nvmlInit_v2")?) };
    let count_fn: Count = unsafe { std::mem::transmute(sym("nvmlDeviceGetCount_v2")?) };
    let by_index: ByIndex = unsafe { std::mem::transmute(sym("nvmlDeviceGetHandleByIndex_v2")?) };
    let get_name: GetName = unsafe { std::mem::transmute(sym("nvmlDeviceGetName")?) };
    let get_energy: GetEnergy =
        unsafe { std::mem::transmute(sym("nvmlDeviceGetTotalEnergyConsumption")?) };
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
        let mut buf = [0 as std::ffi::c_char; 96];
        let name = if unsafe { get_name(handle, buf.as_mut_ptr(), buf.len() as u32) } == 0 {
            let bytes: Vec<u8> = buf
                .iter()
                .copied()
                .take_while(|c| *c != 0)
                .map(|c| c as u8)
                .collect();
            String::from_utf8_lossy(&bytes).into_owned()
        } else {
            format!("gpu{i}")
        };
        names.push(name.clone());
        devices.push(NvmlDevice { handle, name });
    }
    // Thor's NVML builds the symbol and then returns NOT_SUPPORTED. A later read would
    // fail the same way, so the open is the failure and the module-input meter is the fallback.
    let mut probe_mj = 0u64;
    let rc = unsafe { get_energy(devices[0].handle, &mut probe_mj) };
    if rc == 3 {
        unsafe { dlclose(lib) };
        return Err(
            "nvmlDeviceGetTotalEnergyConsumption returned NOT_SUPPORTED. CUDA energy uses the module-input VIN meter, not this counter."
                .into(),
        );
    }
    if rc != 0 {
        unsafe { dlclose(lib) };
        return Err(format!("nvmlDeviceGetTotalEnergyConsumption returned {rc}"));
    }
    Ok(Nvml {
        lib,
        get_energy,
        devices,
        names,
    })
}

#[cfg(unix)]
unsafe extern "C" {
    fn dlopen(filename: *const std::ffi::c_char, flags: i32) -> *mut std::ffi::c_void;
    fn dlsym(
        handle: *mut std::ffi::c_void,
        symbol: *const std::ffi::c_char,
    ) -> *mut std::ffi::c_void;
    fn dlclose(handle: *mut std::ffi::c_void) -> i32;
}

/// Millijoule counter delta, wrap included, as joules.
pub fn mj_delta_j(prev: u64, now: u64) -> f64 {
    let d = if now >= prev {
        now - prev
    } else {
        now.wrapping_add(u64::MAX - prev).wrapping_add(1)
    };
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
pub fn idle_baseline(
    meter: &mut ferrotherm_meter::Meter,
) -> Result<ferrotherm_meter::Baseline, String> {
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
        assert!(
            !ok,
            "a negative incremental is not attributable even when it clears the band"
        );
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
    fn amd_rapl_is_not_intel_rapl_and_neither_is_wall() {
        assert_eq!(cpu_vendor_from("vendor_id : AuthenticAMD\n"), "amd");
        assert_eq!(cpu_vendor_from("vendor_id : GenuineIntel\n"), "intel");
        assert_eq!(cpu_vendor_from("model name : nothing\n"), "unknown");
        let amd = rapl_basis("amd");
        assert!(amd.contains("not wall power"), "{amd}");
        assert!(amd.contains("no DRAM domain"), "{amd}");
        assert!(amd.contains("the energy data is modeled, not measured"), "{amd}");
        assert!(amd.contains("arXiv:2108.00808"), "{amd}");
        assert!(amd.contains("has not been validated"), "{amd}");
        assert!(amd.contains("does not claim that AMD documentation says"), "{amd}");
        assert!(amd.contains("wall-meter cross-check"), "{amd}");
        let intel = rapl_basis("intel");
        assert!(intel.contains("not wall power"), "{intel}");
        assert!(intel.contains("not the AMD model"), "{intel}");
        assert!(!intel.contains("modeled, not measured"), "{intel}");
    }

    #[test]
    fn rapl_on_the_named_framework_is_primary_and_macmon_is_secondary() {
        assert_eq!(energy_rank("rapl", "framework-128gb-0000"), "primary");
        assert_eq!(energy_rank("rapl", "framework-32gb-0001"), "unranked");
        assert_eq!(
            energy_rank("macmon", "Davids-MacBook-Pro-M5-Max-48GB-4.local"),
            "secondary"
        );
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
            let j = trial_joules(
                inc + 10.0,
                10.0,
                0.01,
                1.0,
                "rapl",
                "package",
                g,
                MIN_COUNTER_UPDATES,
                true,
            );
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
        let quiet: Vec<_> = (0..5)
            .map(|i| mk(GRADE_QUIET, 1.0 + i as f64 * 0.1))
            .collect();
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
        assert_eq!(
            count_counter_updates(&[1.0, 2.0, 2.0, 3.0, f64::NAN, 4.0]),
            3
        );
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
        assert!(
            !one_tick.attributable,
            "one counter tick is below the cadence floor"
        );

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
            let j = trial_joules(
                2.0,
                1.0,
                0.01,
                1.0,
                if scope == "package" { "rapl" } else { "macmon" },
                scope,
                GRADE_CONTENDED,
                MIN_COUNTER_UPDATES,
                true,
            );
            EnergyTrial::from_parts(
                workload,
                "ferrotherm",
                0,
                4,
                MIN_COUNTER_UPDATES,
                2.0,
                j,
                10.0,
                Some(2.5),
                Some(0.2),
                GRADE_CONTENDED,
                "primary",
            )
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
        let quiet = quiet_gate_from(
            &[Some(1.0), Some(QUIET_LOAD), Some(0.5)],
            &[(1, 0.1)],
            true,
            60.0,
        );
        assert!(quiet.held);
        assert_eq!(quiet.max_load1, Some(QUIET_LOAD));
        assert_eq!(quiet.other_core_busy, vec![(1, 0.1)]);
        let over = quiet_gate_from(&[Some(1.0), Some(QUIET_LOAD + 0.01)], &[], true, 60.0);
        assert!(
            !over.held,
            "one sample over the unchanged gate refuses the run"
        );
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
    fn the_bundle_is_quiet_only_when_the_energy_phase_is_quiet() {
        assert_eq!(bundle_grade(GRADE_QUIET), GRADE_QUIET);
        assert_eq!(bundle_grade("load_gated"), "load_gated");
        assert_eq!(bundle_grade(GRADE_CONTENDED), GRADE_CONTENDED);
        assert_eq!(bundle_grade("unavailable"), "load_gated");
        // A passing preflight and a quiet accuracy phase do not promote a refused energy phase.
        assert_ne!(bundle_grade("load_gated"), GRADE_QUIET);
        assert_eq!(ENERGY_WATCH_INTERVAL, Duration::from_secs(1));
        assert_eq!(QUIET_LOAD, 2.0);
    }

    #[test]
    fn an_energy_watch_aborts_on_the_first_sample_over_the_threshold() {
        let held = grade_energy_watch(&[Some(1.0), Some(QUIET_LOAD), Some(1.5)]);
        assert_eq!(held.evidence_grade, GRADE_QUIET);
        assert!(held.abort.is_none());
        assert_eq!(held.samples.len(), 3);

        let refused = grade_energy_watch(&[Some(1.0), Some(1.5), Some(2.4), Some(1.0)]);
        assert_eq!(refused.evidence_grade, "load_gated");
        let abort = refused.abort.expect("the 2.4 sample");
        assert!((abort.load1 - 2.4).abs() < 1e-12);
        assert_eq!(abort.at_unix_s, 2.0);
        assert_eq!(
            refused.samples.len(),
            3,
            "samples after the abort are not a recovery"
        );
        assert!(grade_energy_watch(&[]).evidence_grade != GRADE_QUIET);
        assert_eq!(grade_energy_watch(&[None]).evidence_grade, "load_gated");
    }

    #[test]
    fn a_settle_needs_sixty_consecutive_samples_and_a_high_sample_resets_it() {
        let mut quiet = vec![Some(3.0)];
        quiet.extend(std::iter::repeat_n(Some(1.5), 59));
        let short = settle_outcome(&quiet, SETTLE_STREAK, SETTLE_BOUND);
        assert!(!short.settled, "{}", short.reason);
        quiet.push(Some(1.2));
        let ok = settle_outcome(&quiet, SETTLE_STREAK, SETTLE_BOUND);
        assert!(ok.settled, "{}", ok.reason);
        assert_eq!(ok.samples.len(), 61);
        let mut bounded = vec![Some(1.0); 59];
        bounded.push(Some(2.01));
        bounded.extend(std::iter::repeat_n(Some(1.0), 59));
        let reset = settle_outcome(&bounded, SETTLE_STREAK, 60);
        assert!(!reset.settled);
        assert!(reset.reason.contains("60"));
        assert_eq!(SETTLE_STREAK, 60);
        const _: () = assert!(SETTLE_BOUND >= SETTLE_STREAK);
        assert_eq!(QUIET_LOAD, 2.0);
    }

    #[test]
    fn a_bracket_sample_above_two_refuses_and_an_empty_bracket_is_not_quiet() {
        assert!(bracket_load_offender(&[Some(1.0), Some(2.0)]).is_none());
        assert_eq!(bracket_load_offender(&[Some(1.0), Some(2.41)]), Some(2.41));
        assert!(bracket_load_offender(&[]).is_some());
    }

    #[test]
    fn linux_other_core_and_macos_foreign_cpu_refuse_on_mean_or_sample() {
        let rule = TrialWindowRule::defaults();
        assert!(rule.enforce);
        assert!((rule.other_core_mean_max - 0.05).abs() < 1e-12);
        assert!((rule.other_core_sample_max - 0.15).abs() < 1e-12);
        assert!((rule.foreign_cpu_mean_max - 0.15).abs() < 1e-12);
        assert!((rule.foreign_cpu_sample_max - 0.50).abs() < 1e-12);
        assert!(
            fraction_refusal(
                &[0.04, 0.04],
                rule.other_core_mean_max,
                rule.other_core_sample_max
            )
            .is_none()
        );
        assert_eq!(
            fraction_refusal(
                &[0.04, 0.16],
                rule.other_core_mean_max,
                rule.other_core_sample_max
            ),
            Some("a sample exceeded the maximum")
        );
        assert_eq!(
            fraction_refusal(
                &[0.06, 0.06],
                rule.other_core_mean_max,
                rule.other_core_sample_max
            ),
            Some("the mean exceeded the maximum")
        );
        assert_eq!(
            fraction_refusal(&[], rule.other_core_mean_max, rule.other_core_sample_max),
            Some("the trial window recorded no finite sample")
        );
        assert!(
            fraction_refusal(
                &[0.10, 0.14],
                rule.foreign_cpu_mean_max,
                rule.foreign_cpu_sample_max
            )
            .is_none()
        );
        assert!(
            fraction_refusal(
                &[0.10, 0.51],
                rule.foreign_cpu_mean_max,
                rule.foreign_cpu_sample_max
            )
            .is_some()
        );
        assert!(
            fraction_refusal(
                &[0.20, 0.20],
                rule.foreign_cpu_mean_max,
                rule.foreign_cpu_sample_max
            )
            .is_some()
        );
        assert!(linux_quiet_needs_pin("linux", 0));
        assert!(!linux_quiet_needs_pin("linux", 4));
        assert!(!linux_quiet_needs_pin("macos", 0));
        let (mean, sigma, _, drift_ok) = combine_bracket(10.0, 0.1, 12.0, 0.1);
        assert!(!drift_ok);
        assert!((mean - 11.0).abs() < 1e-9);
        assert!(sigma > 0.0);
        assert!(macos_drift_refuses_quiet("macos", false));
        assert!(!macos_drift_refuses_quiet("macos", true));
        assert!(!macos_drift_refuses_quiet("linux", false));
    }

    #[test]
    fn foreign_cpu_attributes_other_processes_and_does_not_price_them() {
        assert!((parse_cpu_clock("1:02.5").unwrap() - 62.5).abs() < 1e-9);
        assert!((parse_cpu_clock("1-00:00:01").unwrap() - 86401.0).abs() < 1e-9);
        let row = parse_ps_cpu_row("10 1 0:01").unwrap();
        assert_eq!(row.0, 10);
        let before = [(1u32, 0u32, 0.0), (10, 1, 1.0), (11, 10, 1.0), (20, 2, 5.0)];
        let after = [(1, 0, 0.0), (10, 1, 1.1), (11, 10, 1.4), (20, 2, 5.5)];
        // self is 10. 11 is its child. 20 is foreign and gained 0.5 s over 1 s of wall.
        let frac = foreign_cpu_fraction(&before, &after, 10, 1.0).unwrap();
        assert!((frac - 0.5).abs() < 1e-9, "{frac}");
        let joules = trial_joules(
            1.0,
            0.2,
            0.01,
            1.0,
            "macmon",
            "soc-telemetry",
            GRADE_QUIET,
            MIN_COUNTER_UPDATES,
            true,
        );
        assert!(
            (joules.gross.unwrap() - 1.0).abs() < 1e-12,
            "a cpu fraction is not subtracted from joules"
        );
    }

    #[test]
    fn a_contended_host_lock_is_refused_and_a_free_lock_records_the_holder() {
        // A second `flock` in this process succeeds on macOS: the lock is per process there.
        // The holder is a child, which is the contention the receipt has to refuse.
        if std::env::var_os("FT_HOLD_LOCK").is_some() {
            let path = std::env::var("FT_HOLD_LOCK_PATH").expect("path");
            let ready = std::env::var("FT_HOLD_LOCK_READY").expect("ready");
            let held = HostLock::acquire(std::path::Path::new(&path), 2.0).expect("child acquires");
            std::fs::write(&ready, held.holder_pid.to_string()).expect("ready");
            std::thread::sleep(Duration::from_secs(2));
            drop(held);
            return;
        }
        assert_eq!(HOST_LOCK_DEFAULT, "/tmp/bmi-energy.lock");
        assert_eq!(HOST_LOCK_WAIT_S, 3600.0);
        assert_eq!(format_unix_utc(0.0), "1970-01-01T00:00:00.000Z");
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "ft-bmi-energy-{}-{}.lock",
            std::process::id(),
            line!()
        ));
        let ready = dir.join(format!(
            "ft-bmi-energy-{}-{}.ready",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&ready);
        let exe = std::env::current_exe().expect("test exe");
        let mut child = std::process::Command::new(exe)
            .args(["--exact", "energy::tests::a_contended_host_lock_is_refused_and_a_free_lock_records_the_holder"])
            .env("FT_HOLD_LOCK", "1")
            .env("FT_HOLD_LOCK_PATH", &path)
            .env("FT_HOLD_LOCK_READY", &ready)
            .spawn()
            .expect("spawn holder");
        let start = std::time::Instant::now();
        while !ready.exists() {
            if start.elapsed() > Duration::from_secs(8) {
                let _ = child.kill();
                panic!("holder did not acquire");
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let err = match HostLock::acquire(&path, 0.5) {
            Err(text) => text,
            Ok(lock) => {
                drop(lock);
                panic!("contended lock was acquired");
            }
        };
        assert_eq!(err, HOST_LOCK_REFUSAL);
        let _ = child.wait();
        let again = HostLock::acquire(&path, 2.0).expect("released");
        assert_eq!(again.holder_pid, std::process::id());
        assert!(again.acquired_utc().ends_with('Z'));
        let released = again.release();
        assert!(released.ends_with('Z'));
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&ready);
    }

    #[test]
    fn tegrastats_vdd_in_is_one_rail_and_other_rails_are_not_summed() {
        let line =
            "RAM 1543/7772MB CPU [1%@102] VDD_IN 1864/1864 VDD_CPU_GPU_CV 341/341 VDD_SOC 717/717";
        let parsed = parse_tegrastats_module_input(line).unwrap();
        assert_eq!(parsed.label, "VDD_IN");
        assert!((parsed.watts - 1.864).abs() < 1e-12);
        let mw = "VDD_IN 2500mW/2500mW VDD_CPU_GPU_CV 100mW/100mW";
        assert!((parse_tegrastats_module_input(mw).unwrap().watts - 2.5).abs() < 1e-12);
        assert!(parse_tegrastats_module_input("VDD_CPU_GPU_CV 341/341 VDD_SOC 717/717").is_none());
        assert!(parse_tegrastats_module_input("no power here").is_none());
    }

    #[test]
    fn thor_tegrastats_vin_is_module_input_and_subrails_are_not_added() {
        // R39 tegrastats power-block form. VIN is the module-input token. VIN_SYS_5V0 is a
        // different token and must not be parsed as VIN. The value is the first number, in mW.
        let line = "11-09-2026 21:04:11 RAM 4512/125772MB (lfb 2048x4MB) SWAP 0/0MB (cached 0MB) \
                    CPU [2%@729,1%@729] EMC_FREQ 3%@3199 GR3D_FREQ 0%@[0,0,0] \
                    VIN 26140mW/25980mW VDD_CPU_SOC_MSS 4120mW/4001mW VDD_GPU 890mW/870mW \
                    VIN_SYS_5V0 1510mW/1490mW";
        let sample = parse_tegrastats_module_input(line).expect("VIN");
        assert_eq!(sample.label, "VIN");
        assert!((sample.watts - 26.140).abs() < 1e-12, "{}", sample.watts);
        let summed = 26.140 + 4.120 + 0.890 + 1.510;
        assert!(
            (sample.watts - summed).abs() > 1.0,
            "sub-rails were added into VIN"
        );
        let bare = parse_tegrastats_module_input("VIN 18000/17900 VDD_GPU 100/100").unwrap();
        assert_eq!(bare.label, "VIN");
        assert!((bare.watts - 18.0).abs() < 1e-12);
        assert!(
            parse_tegrastats_module_input("VDD_GPU 890mW/870mW VIN_SYS_5V0 1510mW/1490mW")
                .is_none()
        );
    }

    #[test]
    fn thor_ina238_hwmon_is_vin_and_ina3221_channels_stay_separate() {
        let root = std::env::temp_dir().join(format!("ft-thor-hwmon-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let ina = root.join("2-0044/hwmon/hwmon0");
        std::fs::create_dir_all(&ina).unwrap();
        std::fs::write(ina.join("name"), "ina238\n").unwrap();
        std::fs::write(ina.join("power1_input"), "26000000\n").unwrap();
        std::fs::write(ina.join("in0_input"), "28000\n").unwrap();
        let found = find_ina238(&root).expect("ina238");
        assert!(found.verified, "{}", found.note);
        assert_eq!(found.i2c, "2-0044");
        assert_eq!(found.name, "ina238");
        assert_eq!(found.source, "power1_input");
        assert!(!found.counter);
        assert!(found.note.contains("not wall"), "{}", found.note);
        assert!(found.note.contains("28.000 V"), "{}", found.note);
        let raw: f64 = std::fs::read_to_string(&found.path)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        assert!((raw * 1.0e-6 - 26.0).abs() < 1e-9);

        let other = root.join("1-0044/hwmon/hwmon1");
        std::fs::create_dir_all(&other).unwrap();
        std::fs::write(other.join("name"), "ina238\n").unwrap();
        std::fs::write(other.join("energy1_input"), "5000000\n").unwrap();
        std::fs::write(other.join("power1_input"), "1000000\n").unwrap();
        // The verified 2-0044 node is preferred over this unverified one.
        let again = find_ina238(&root).unwrap();
        assert!(again.verified);
        assert_eq!(again.i2c, "2-0044");
        let _ = std::fs::remove_dir_all(&ina);
        let unverified = find_ina238(&root).unwrap();
        assert!(!unverified.verified, "{}", unverified.note);
        assert!(
            unverified.note.contains("unverified"),
            "{}",
            unverified.note
        );
        assert_eq!(unverified.source, "energy1_input");
        assert!(unverified.counter);

        let rails = root.join("2-0040/hwmon/hwmon2");
        std::fs::create_dir_all(&rails).unwrap();
        for (i, (label, volt, curr)) in [
            ("VDD_GPU", "800", "1000"),
            ("VDD_CPU_SOC_MSS", "800", "5000"),
            ("VIN_SYS_5V0", "5000", "300"),
        ]
        .into_iter()
        .enumerate()
        {
            let n = i + 1;
            std::fs::write(rails.join(format!("in{n}_label")), format!("{label}\n")).unwrap();
            std::fs::write(rails.join(format!("in{n}_input")), format!("{volt}\n")).unwrap();
            std::fs::write(rails.join(format!("curr{n}_input")), format!("{curr}\n")).unwrap();
        }
        let listed = ferrotherm_meter::ina3221::Rails::list_at(&rails).unwrap();
        let scopes: Vec<String> = listed
            .iter()
            .filter_map(|rail| subrail_scope(&rail.label))
            .collect();
        assert_eq!(scopes.len(), 3);
        assert!(scopes.iter().all(|s| s.starts_with("subrail:")));
        assert!(scopes.iter().all(|s| s != "vin" && s != "board"));
        assert!(!scopes_may_share_a_table("vin", "subrail:VDD_GPU"));
        assert!(scopes_may_share_a_table(
            "subrail:VDD_GPU",
            "subrail:VDD_GPU"
        ));
        assert!(!scopes_may_share_a_table(
            "subrail:VDD_GPU",
            "subrail:VIN_SYS_5V0"
        ));
        assert!(subrail_scope("sum of shunt voltages").is_none());
        assert!(subrail_scope("VDD_IN").is_none());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn an_ina238_behind_hwmon_symlinks_is_module_input_not_wall() {
        let root = std::env::temp_dir().join(format!("ft-thor-symlink-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        // Deeper than the walk bound, so a directory-only scan never sees `name`.
        let real = root.join("deep/a/b/c/d/e/f/2-0044/hwmon/hwmon7");
        std::fs::create_dir_all(&real).unwrap();
        std::fs::write(real.join("name"), "ina238\n").unwrap();
        std::fs::write(real.join("power1_input"), "31000000\n").unwrap();
        std::fs::write(real.join("in0_input"), "20000\n").unwrap();
        let class = root.join("class/hwmon");
        let i2c = root.join("bus/i2c/devices");
        std::fs::create_dir_all(&class).unwrap();
        std::fs::create_dir_all(&i2c).unwrap();
        std::os::unix::fs::symlink(&real, class.join("hwmon7")).unwrap();
        std::os::unix::fs::symlink(real.parent().unwrap().parent().unwrap(), i2c.join("2-0044")).unwrap();
        let found = find_ina238(&root).expect("symlink");
        assert!(found.verified, "{}", found.note);
        assert_eq!(found.i2c, "2-0044");
        assert_eq!(found.source, "power1_input");
        assert!(found.note.contains("backend chosen: ina238"), "{}", found.note);
        assert!(found.note.contains("module input"), "{}", found.note);
        assert!(found.note.contains("not wall"), "{}", found.note);
        assert!(found.note.contains("class/hwmon"), "{}", found.note);
        let raw: f64 = std::fs::read_to_string(&found.path).unwrap().trim().parse().unwrap();
        assert!((raw * 1.0e-6 - 31.0).abs() < 1e-9, "{raw}");
        std::fs::remove_file(class.join("hwmon7")).unwrap();
        let via_i2c = find_ina238(&root).expect("i2c symlink");
        assert_eq!(via_i2c.i2c, "2-0044");
        assert!(via_i2c.note.contains("bus/i2c/devices"), "{}", via_i2c.note);
        assert!(via_i2c.note.contains("not wall"), "{}", via_i2c.note);
        std::fs::remove_file(i2c.join("2-0044")).unwrap();
        assert!(find_ina238(&root).is_err(), "the deep path alone is not the discovery");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_negative_or_drifting_bracket_is_not_attributable_and_pairs_keep_their_sign() {
        let neg = trial_joules(
            4.0,
            4.0,
            0.01,
            2.0,
            "rapl",
            "package",
            GRADE_CONTENDED,
            MIN_COUNTER_UPDATES,
            true,
        );
        assert!(neg.incremental.unwrap() < 0.0);
        assert!(!neg.attributable);
        assert!(joules_per_ess(&neg, 100.0).is_none());
        let drifted = trial_joules(
            20.0,
            4.0,
            0.01,
            2.0,
            "rapl",
            "package",
            GRADE_CONTENDED,
            MIN_COUNTER_UPDATES,
            false,
        );
        assert!(!drifted.attributable);
        let (mean, sigma, drift, ok) = combine_bracket(10.0, 0.2, 10.1, 0.2);
        assert!((mean - 10.05).abs() < 1e-12);
        assert!(
            ok,
            "0.1 W drift against sigma 0.2 is inside 3 sigma, drift {drift} sigma {sigma}"
        );
        let (_, _, big, ok) = combine_bracket(10.0, 0.1, 12.0, 0.1);
        assert!(!ok && (big - 2.0).abs() < 1e-12);
        let delta = paired_delta(1.5, 4.0);
        assert!((delta + 2.5).abs() < 1e-12);
        assert!(paired_attributable(
            delta,
            0.05,
            2.0,
            2.0,
            true,
            MIN_COUNTER_UPDATES,
            MIN_COUNTER_UPDATES
        ));
        assert!(!paired_attributable(
            0.01,
            0.05,
            2.0,
            2.0,
            true,
            MIN_COUNTER_UPDATES,
            MIN_COUNTER_UPDATES
        ));
        assert!(!paired_attributable(
            delta,
            0.05,
            2.0,
            2.0,
            false,
            MIN_COUNTER_UPDATES,
            MIN_COUNTER_UPDATES
        ));
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
