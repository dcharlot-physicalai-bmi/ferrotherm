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
//! macmon whole-system energy do not share a table.
//!
//! RAPL on hostname `framework-128gb-0000` is the primary energy evidence. That host
//! reads package `energy_uj` without root. The permission denial was `framework-32gb-0001`.
//! macmon is secondary and whole-system.
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

/// `gross - idle * seconds`, and whether that delta clears `3 * sigma * seconds`.
///
/// A delta inside the band is still returned. `attributable` says whether it may be
/// divided into a per-sample figure. Dividing a number that did not clear the baseline's
/// own wander reports precision the counter did not have.
pub fn split_incremental(gross: f64, idle_watts: f64, idle_sigma: f64, seconds: f64) -> (f64, bool) {
    let incremental = gross - idle_watts * seconds;
    let band = 3.0 * idle_sigma * seconds;
    let attributable = incremental.abs() > band && seconds > 0.0 && idle_sigma.is_finite();
    (incremental, attributable)
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
pub fn sample_during<R>(
    mut read: impl FnMut() -> Option<f64> + Send,
    body: impl FnOnce() -> R,
) -> (Vec<f64>, f64, R) {
    let stop = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|scope| {
        let flag = &stop;
        let handle = scope.spawn(move || {
            let mut samples = Vec::new();
            while !flag.load(std::sync::atomic::Ordering::Relaxed) {
                if let Some(x) = read() {
                    samples.push(x);
                }
                std::thread::sleep(Duration::from_micros(250));
            }
            samples
        });
        let _stop_on_drop = StopReader(&stop);
        let started = std::time::Instant::now();
        let value = body();
        let seconds = started.elapsed().as_secs_f64();
        drop(_stop_on_drop);
        let samples = handle.join().unwrap_or_default();
        (samples, seconds, value)
    })
}

struct StopReader<'a>(&'a std::sync::atomic::AtomicBool);

impl Drop for StopReader<'_> {
    fn drop(&mut self) {
        self.0.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Package energy and whole-system energy are different measurements.
///
/// The labels that may be summarised are `package` (RAPL package or `power/energy-pkg`),
/// `whole-system` (macmon), `psys`, `board` (INA3221), and `gpu` (NVML). Two rows share
/// a scope only when the label is the same one of those.
pub fn scopes_may_share_a_table(a: &str, b: &str) -> bool {
    a == b && matches!(a, "package" | "whole-system" | "psys" | "board" | "gpu")
}

/// A, B, A, B, ... with `n` runs of each sampler.
///
/// The sequence begins ferrotherm, THRML, ferrotherm. Neither sampler is a block at the end.
pub fn ab_schedule(n: usize) -> Vec<&'static str> {
    (0..n * 2).map(|i| if i % 2 == 0 { "ferrotherm" } else { "thrml" }).collect()
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
) -> Joules {
    let (incremental, clears_band) = split_incremental(gross, idle_watts, idle_sigma, seconds);
    let cadence = counter_updates >= MIN_COUNTER_UPDATES;
    let moved = gross.is_finite() && gross > 0.0;
    let attributable = clears_band && cadence && moved && incremental.is_finite();
    let note = if !cadence {
        format!(
            "attribution refused: the counter advanced {counter_updates} times and the floor is {MIN_COUNTER_UPDATES}"
        )
    } else if !moved {
        "attribution refused: gross energy was not strictly positive".to_string()
    } else if !clears_band {
        "incremental joules are inside the idle band and are not a per-sample price".to_string()
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
        }
    }
}

/// Whether two trials are the same cell of a table.
///
/// A cell is one workload, one sampler, one instrument, one scope, and one evidence grade.
/// Package joules and whole-system joules are not a cell.
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
    })
}

/// One summary per workload, sampler, instrument, and scope, in first-seen order.
///
/// Package rows and whole-system rows stay in different summaries.
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
    pub perf: Option<PerfPkg>,
    pub perf_note: String,
    pub nvml: Option<Nvml>,
    pub nvml_note: String,
    pub hostname: String,
    pub macos_policy: &'static str,
}

impl Probe {
    pub fn detect() -> Probe {
        let load1 = load_average();
        let quiet = load1.is_some_and(|l| l <= QUIET_LOAD);
        let wall = Meter::detect().filter(|m| m.backend() != "rapl");
        let wall_note = match &wall {
            Some(m) => format!("backend={} machine={} scope={:?}", m.backend(), m.machine(), m.scope()),
            None => "no macmon and no INA3221 rail".into(),
        };
        let (rapl, rapl_note) = match RaplCounter::open() {
            Ok(c) => {
                let note = format!("{} ({})", c.name, c.kind);
                (Some(c), note)
            }
            Err(e) => (None, e),
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
                let note = format!("{} device(s): {}", n.devices.len(), n.names.join(", "));
                (Some(n), note)
            }
            Err(e) => (None, e),
        };
        Probe {
            load1,
            quiet,
            wall,
            wall_note,
            rapl,
            rapl_note,
            perf,
            perf_note,
            nvml,
            nvml_note,
            hostname: read_hostname(),
            macos_policy: "macOS power, when that OS is the host, is ferrotherm_meter::Meter::macmon \
                           (rootless IOReport via the macmon binary). This harness has no other macOS path.",
        }
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
        let n = unsafe { read(self.fd, buf.as_mut_ptr(), buf.len()) };
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
    let n = unsafe { read(fd, buf.as_mut_ptr(), buf.len()) };
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
    fn read(fd: i32, buf: *mut u8, count: usize) -> isize;
    fn close(fd: i32) -> i32;
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
        let seq = ab_schedule(5);
        assert_eq!(&seq[..3], &["ferrotherm", "thrml", "ferrotherm"]);
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
            let j = trial_joules(inc + 10.0, 10.0, 0.01, 1.0, "rapl", "package", g, MIN_COUNTER_UPDATES);
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
        );
        assert!(held.attributable);
        assert!(joules_per_ess(&held, 1000.0).is_some());
    }

    #[test]
    fn package_and_whole_system_are_not_one_table_and_workloads_are_not_pooled() {
        assert!(!scopes_may_share_a_table("package", "whole-system"));
        assert!(scopes_may_share_a_table("package", "package"));
        assert!(!scopes_may_share_a_table("cpu-package", "cpu-package"));
        let row = |workload: &'static str, scope: &'static str| {
            let j = trial_joules(2.0, 1.0, 0.01, 1.0, if scope == "package" { "rapl" } else { "macmon" }, scope, GRADE_CONTENDED, MIN_COUNTER_UPDATES);
            EnergyTrial::from_parts(workload, "ferrotherm", 0, 4, MIN_COUNTER_UPDATES, 2.0, j, 10.0, Some(2.5), Some(0.2), GRADE_CONTENDED, "primary")
        };
        let package = row("thrml_readme_ising_chain_n5", "package");
        let whole = row("thrml_readme_ising_chain_n5", "whole-system");
        let other = row("ising_grid2d_open_3x3", "package");
        assert!(!may_share_summary(&package, &whole));
        assert!(!may_share_summary(&package, &other));
        assert!(summarise_one_grade(&[package.clone(), whole.clone()]).is_err());
        assert!(summarise_one_grade(&[package.clone(), other.clone()]).is_err());
        let sums = summarise_workloads(&[package, whole, other]);
        assert_eq!(sums.len(), 3);
        assert_eq!(sums[0].scope, "package");
        assert_eq!(sums[0].workload, "thrml_readme_ising_chain_n5");
        assert_eq!(sums[1].scope, "whole-system");
        assert_eq!(sums[1].instrument, "macmon");
        assert_eq!(sums[2].workload, "ising_grid2d_open_3x3");
    }
}
