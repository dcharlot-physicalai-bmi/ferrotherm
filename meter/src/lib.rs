//! Measured power in the backend's own scope, so a joules figure can describe the machine that
//! produced it.
//!
//! `ferrotherm`'s ledger counts operations exactly and prices them against a [`Prices`] table. The
//! only table in the tree is `Z1_SPICE` — pre-silicon SPICE estimates for a device that has not
//! been characterised — and until recently every fabric declared it, so a laptop's Gibbs sweeps
//! reported another company's unfabricated accelerator's energy. `Prices::UNSTATED` made that
//! honest. This crate makes it unnecessary: measure the machine you are on, and derive the prices
//! from the measurement.
//!
//! # What it measures, and what it subtracts
//!
//! Each backend has a scope. `macmon` `sys_power` is `SoC` telemetry (the CPU, the integrated GPU,
//! and the ANE). It is not wall power: the display, the fans, and the rest of the board are
//! outside it. RAPL package energy is the CPU package. An INA3221 rail is the board rail that
//! chip sits on. The integral is a trapezoid over the timestamped samples, with the window edges
//! placed explicitly. Then **idle is subtracted**: a machine that draws 20 W doing nothing does
//! not charge that to your workload, and a "measurement" that forgets to subtract it reports
//! mostly the cost of the computer being switched on. Both numbers are reported so the subtraction
//! is visible rather than assumed.
//!
//! # What it refuses
//!
//! A workload shorter than a few sampling intervals. The backend reports power on a fixed tick, so
//! a 50 ms run sampled every 200 ms collects zero or one sample — and one sample of a fluctuating
//! quantity is not an estimate of its mean. [`Meter::measure`] returns an error naming the shortfall
//! rather than a number computed from too little.
//!
//! # Backends
//!
//! - **macOS**: `macmon pipe`, which reads the `SoC`'s own power counters. Detected on `PATH`.
//! - **Jetson / Linux**: [`ina3221`], the shunt monitors on the board. Rail discovery, label
//!   matching, both driver layouts and both unit conversions are tested against fixture
//!   directories; **no reading has been taken from real hardware**, because the Jetson on this
//!   tailnet is offline. That split is deliberate and stated rather than papered over -- the
//!   arithmetic that was going to be wrong is tested, and the hardware path is not.
//!
//! ```no_run
//! # use ferrotherm::{gibbs::Sampler, ising::lattice2d, ledger::Ledger};
//! # fn main() -> Result<(), String> {
//! let mut meter = ferrotherm_meter::Meter::detect().ok_or("no power backend on this machine")?;
//! let idle = meter.idle(std::time::Duration::from_secs(2))?;
//!
//! let g = lattice2d(64, 1.0);
//! let mut led = Ledger::default();
//! let run = meter.measure(idle, || {
//!     let mut s = Sampler::new(&g, 0.7, 1);
//!     s.sweeps(2_000, Some(&mut led));
//! })?;
//!
//! println!("{:.3} J above idle over {:.2} s", run.joules_above_idle, run.seconds);
//! let prices = run.prices_from(&led)?;   // measured, and it says so
//! # Ok(()) }
//! ```

mod rapl;

use ferrotherm::ledger::{Ledger, Prices};
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Fewer than this many power samples inside a window and the mean is not an estimate.
const MIN_SAMPLES: usize = 8;

/// How often the backend is asked for a reading.
const INTERVAL_MS: u64 = 100;

/// How long to let the machine quieten before taking a baseline.
///
/// Three seconds, not one: measured on an M5 Max, a baseline taken immediately after a heavy run
/// read 68.5 W while the run it followed had averaged 64.1 W. Fans and thermal management lag the
/// workload, so "idle" straight after load is systematically HIGHER than true idle -- and
/// subtracting it makes the next run look free. Machines with moving parts need settling time and
/// the protocol has to include it.
const SETTLE: Duration = Duration::from_secs(3);

// The load-average instrument and its threshold now live in the core crate, at
// `ferrotherm::host`. They started here, guarding an idle POWER baseline. Then a benchmark harness
// reported 85.7 s for a G1 search that takes about 14 s on a quiet machine, and it turned out the
// timing side of the workspace had no equivalent guard -- the same failure, in a crate that cannot
// depend on this one. Rather than a second copy, the reading moved down and both sides import it.
//
// `QUIET_LOAD` is two, not a fraction of the core count, and the reason is what a load average
// counts: RUNNABLE THREADS, not utilisation. Two threads that never sleep are two cores' worth of
// heat whether the machine has four cores or forty, and that heat lands in the same power reading
// the baseline is made of. A percentage-of-cores threshold would quietly permit an
// 18-core machine to take a "baseline" with four builds running.
use ferrotherm::host::{load_average, QUIET_LOAD};

pub mod ina3221;

/// A source of power readings, in watts, in the backend's own scope.
pub struct Meter {
    /// The backend subprocess, when the backend IS one. `ina3221` reads sysfs from a thread and
    /// has none -- so this became an Option rather than forcing a fake child to exist.
    child: Option<Child>,
    /// Readings with the instant they ARRIVED, filled by a reader thread.
    ///
    /// A thread rather than reading inline, because the backend ticks on its own clock and a read
    /// blocks until the next tick. Reading inline meant a "drain the buffer" step could block for
    /// seconds, and a workload's window could only be defined by what happened to be buffered
    /// rather than by when it ran. With arrival instants the window is a time interval, which is
    /// what it always should have been.
    readings: Arc<Mutex<Vec<(Instant, f64)>>>,
    machine: String,
    backend: &'static str,
    scope: Scope,
}

/// What a backend's reading actually covers.
///
/// Not decoration. A meter and a device have to be in the same frame for a comparison to mean
/// anything, and they silently were not: RAPL `package-0` on a laptop with a DISCRETE RTX 4050
/// reported the GPU arm of a duty-cycle run as 5.5 W marginal, because the card's draw never enters
/// the CPU package counter. The number was small, plausible, and about the cost of FEEDING the
/// GPU rather than running it. On an Apple `SoC`, `sys_power` covers the integrated GPU and does
/// not cover a discrete card, and it is `SoC` telemetry rather than a wall measurement — so the
/// error appears only when the backend changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    /// A rail or domain that claims everything downstream of it. A discrete card on its own power
    /// feed can still be outside a package counter; this variant is not used for `macmon`.
    WholeSystem,
    /// The CPU package alone. A discrete GPU is **invisible**, and so are RAM, storage and fans.
    CpuPackage,
    /// `macmon` field `sys_power`: telemetry from the `SoC` (CPU, integrated GPU, ANE).
    ///
    /// Not wall power. The display, the fans, and the rest of the board are outside the field.
    /// It does not see a discrete GPU on its own feed.
    SocTelemetry,
}

impl Scope {
    /// Can a run on `device_is_discrete` hardware be attributed with this reading?
    ///
    /// `false` means the answer would be a number about the wrong component, which is worse than
    /// no answer — so callers refuse rather than divide.
    #[must_use = "false means this meter cannot see that device, and reporting anyway attributes the wrong component's energy"]
    pub fn covers(&self, device_is_discrete: bool) -> bool {
        match self {
            Scope::WholeSystem => true,
            Scope::CpuPackage | Scope::SocTelemetry => !device_is_discrete,
        }
    }

    /// Short label stored on a receipt. `soc-telemetry` is `macmon` `sys_power`, not wall power.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Scope::WholeSystem => "whole-system",
            Scope::CpuPackage => "package",
            Scope::SocTelemetry => "soc-telemetry",
        }
    }
}

/// Which backend produced a meter, and which RAPL domain when the backend is RAPL.
fn backend_scope(backend: &str, rapl_kind: &str) -> Scope {
    match (backend, rapl_kind) {
        ("macmon", _) => Scope::SocTelemetry,
        ("rapl", "psys") => Scope::WholeSystem,
        ("rapl", _) => Scope::CpuPackage,
        _ => Scope::WholeSystem,
    }
}

/// A measured idle baseline: its level AND its spread.
///
/// The spread is not decoration. Measured on an M5 Max sitting at ~60 W, two runs of the same
/// workload reported 2.34e-8 and 1.81e-7 J per node update -- an eight-fold spread -- because one
/// of them added about 1 W to a baseline that wanders by more than that on its own. A delta smaller
/// than the baseline's own variation is not a small measurement, it is not a measurement.
#[derive(Clone, Copy, Debug)]
pub struct Baseline {
    /// Time-weighted mean power over the covered baseline window, in the backend's scope.
    pub watts: f64,
    /// Standard deviation of the readings the baseline was computed from.
    pub sigma: f64,
    /// Readings the mean was computed from.
    pub samples: usize,
    /// The machine's 1-minute load average while the baseline was taken, or `None` where it could
    /// not be read.
    ///
    /// Reported rather than merely checked, because it is the one number that says whether this
    /// baseline describes an idle machine or somebody else's build. It is carried on the result so
    /// a figure published from it can be audited later, when the machine is no longer in that state.
    pub load1: Option<f64>,
    /// Median spacing of the samples that entered the baseline integral, in seconds.
    ///
    /// `NaN` when fewer than two samples were available to space.
    pub cadence_s: f64,
}

/// How a window boundary was placed on the sample series.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeFix {
    /// A sample landed on the boundary.
    OnSample,
    /// Power at the boundary is the linear interpolation of the samples on either side.
    Interpolated,
    /// No sample brackets the boundary from outside. The integral stops at the nearest sample
    /// inside the window. The uncovered duration is recorded and is not filled in.
    Uncovered,
}

impl EdgeFix {
    /// Receipt token for this edge.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            EdgeFix::OnSample => "on-sample",
            EdgeFix::Interpolated => "interpolated",
            EdgeFix::Uncovered => "uncovered",
        }
    }
}

/// Trapezoidal integral of a timestamped power series over `[start, end]`.
///
/// Times are seconds on the same clock as the samples. Joules cover only the span whose edges
/// were either on a sample or interpolated. An uncovered edge is omitted from the joules.
#[derive(Clone, Debug)]
pub struct PowerIntegral {
    /// Integral over the covered span, in joules.
    pub joules: f64,
    /// Seconds the trapezoid actually covers.
    pub covered_s: f64,
    /// Seconds the caller asked for.
    pub window_s: f64,
    /// Raw samples whose timestamps fall inside the closed window.
    pub samples: usize,
    /// Median positive spacing of the samples that touched the window, including one bracket
    /// on each side when one exists. `NaN` when fewer than two such samples exist.
    pub cadence_s: f64,
    /// How the start of the window was placed.
    pub left: EdgeFix,
    /// How the end of the window was placed.
    pub right: EdgeFix,
    /// Seconds before the first covered knot.
    pub uncovered_left_s: f64,
    /// Seconds after the last covered knot.
    pub uncovered_right_s: f64,
    /// Time from the start to the nearer sample that brackets it. `NaN` when that edge is uncovered.
    pub left_gap_s: f64,
    /// Time from the end to the nearer sample that brackets it. `NaN` when that edge is uncovered.
    pub right_gap_s: f64,
    /// Absolute power step of the pair that brackets the start. `NaN` when there is no pair.
    pub left_step_w: f64,
    /// Absolute power step of the pair that brackets the end. `NaN` when there is no pair.
    pub right_step_w: f64,
    /// Max minus min power among the knots that were integrated.
    pub power_span_w: f64,
}

impl PowerIntegral {
    /// Bound on the joules the edges could still move, in joules.
    ///
    /// An interpolated edge contributes half its gap times the bracketing power step: the
    /// unsampled stub at the boundary, times the step the two samples already showed. An
    /// uncovered edge contributes its uncovered duration times the power span of the knots.
    /// That span is the range the series actually took, not a fill-in of the missing joules.
    /// An on-sample edge contributes nothing. A `NaN` term means the samples do not state it,
    /// and it stays `NaN` rather than becoming zero.
    #[must_use]
    pub fn cadence_edge_j(&self) -> f64 {
        let left = edge_term(self.left, self.left_gap_s, self.left_step_w, self.uncovered_left_s, self.power_span_w);
        let right = edge_term(self.right, self.right_gap_s, self.right_step_w, self.uncovered_right_s, self.power_span_w);
        left + right
    }
}

fn edge_term(edge: EdgeFix, gap: f64, step: f64, uncovered: f64, span: f64) -> f64 {
    match edge {
        EdgeFix::OnSample => 0.0,
        EdgeFix::Interpolated => {
            if gap.is_finite() && step.is_finite() { 0.5 * gap * step } else { f64::NAN }
        }
        EdgeFix::Uncovered => {
            if uncovered == 0.0 {
                0.0
            } else if span.is_finite() {
                uncovered * span
            } else {
                f64::NAN
            }
        }
    }
}

/// A sample within this many seconds of a boundary counts as on the boundary.
const EDGE_EPS: f64 = 1e-6;

/// Integrate `samples` of `(seconds, watts)` from `start` to `end`.
///
/// The series is sorted by time. A boundary with a sample on it uses that sample. A boundary
/// with a sample on both sides is linearly interpolated. A boundary with no outside sample stops
/// the integral at the nearest interior sample. The missing edge is not replaced by holding the
/// nearest power across it.
///
/// # Errors
///
/// The window is not a positive finite interval, no sample is finite, or the edges leave fewer
/// than two knots.
///
/// # Panics
///
/// Never. A knot list shorter than two is the error above, and the span is read only after that
/// check.
pub fn trapezoid(samples: &[(f64, f64)], start: f64, end: f64) -> Result<PowerIntegral, String> {
    if !start.is_finite() || !end.is_finite() || end <= start {
        return Err(format!("energy window [{start}, {end}] is not a positive finite interval"));
    }
    let mut pts: Vec<(f64, f64)> = samples
        .iter()
        .copied()
        .filter(|(t, p)| t.is_finite() && p.is_finite())
        .collect();
    if pts.is_empty() {
        return Err("no finite power samples".into());
    }
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let on = |t: f64, boundary: f64| (t - boundary).abs() <= EDGE_EPS;
    let (left, p_start, left_gap_s, left_step_w) = place_edge(&pts, start, end, true, on);
    let (right, p_end, right_gap_s, right_step_w) = place_edge(&pts, end, start, false, on);

    let mut knots: Vec<(f64, f64)> = Vec::new();
    if let Some(p) = p_start {
        knots.push((start, p));
    }
    for (t, p) in &pts {
        if *t > start + EDGE_EPS && *t < end - EDGE_EPS {
            knots.push((*t, *p));
        }
    }
    if let Some(p) = p_end {
        knots.push((end, p));
    }
    if knots.len() < 2 {
        return Err(format!(
            "the window [{start}, {end}] has {} knot(s) after placing the edges; a trapezoid needs two",
            knots.len()
        ));
    }
    let mut joules = 0.0;
    for w in knots.windows(2) {
        let dt = w[1].0 - w[0].0;
        if dt > 0.0 {
            joules += 0.5 * (w[0].1 + w[1].1) * dt;
        }
    }
    let first_t = knots[0].0;
    let last_t = knots[knots.len() - 1].0;
    let covered_s = last_t - first_t;
    let uncovered_left_s = if left == EdgeFix::Uncovered { (first_t - start).max(0.0) } else { 0.0 };
    let uncovered_right_s = if right == EdgeFix::Uncovered { (end - last_t).max(0.0) } else { 0.0 };
    let inside: Vec<(f64, f64)> = pts
        .iter()
        .copied()
        .filter(|(t, _)| *t >= start - EDGE_EPS && *t <= end + EDGE_EPS)
        .collect();
    let mut used: Vec<f64> = Vec::new();
    if let Some((t, _)) = pts.iter().rev().find(|(t, _)| *t < start - EDGE_EPS) {
        used.push(*t);
    }
    used.extend(inside.iter().map(|(t, _)| *t));
    if let Some((t, _)) = pts.iter().find(|(t, _)| *t > end + EDGE_EPS) {
        used.push(*t);
    }
    used.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    used.dedup_by(|a, b| (*a - *b).abs() <= EDGE_EPS);
    let mut dts = Vec::new();
    for w in used.windows(2) {
        let dt = w[1] - w[0];
        if dt > 0.0 {
            dts.push(dt);
        }
    }
    let powers: Vec<f64> = knots.iter().map(|k| k.1).collect();
    let (mn, mx) = powers.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), p| (mn.min(*p), mx.max(*p)));
    let power_span_w = if powers.len() >= 2 && mn.is_finite() && mx.is_finite() { mx - mn } else { f64::NAN };
    Ok(PowerIntegral {
        joules,
        covered_s,
        window_s: end - start,
        samples: inside.len(),
        cadence_s: median_positive(&dts),
        left,
        right,
        uncovered_left_s,
        uncovered_right_s,
        left_gap_s,
        right_gap_s,
        left_step_w,
        right_step_w,
        power_span_w,
    })
}

fn place_edge(
    pts: &[(f64, f64)],
    boundary: f64,
    other: f64,
    is_left: bool,
    on: impl Fn(f64, f64) -> bool,
) -> (EdgeFix, Option<f64>, f64, f64) {
    if let Some((_, p)) = pts.iter().find(|(t, _)| on(*t, boundary)) {
        return (EdgeFix::OnSample, Some(*p), 0.0, 0.0);
    }
    let before = pts.iter().rev().find(|(t, _)| *t < boundary - EDGE_EPS).copied();
    let after = pts.iter().find(|(t, _)| *t > boundary + EDGE_EPS).copied();
    if let (Some((t0, p0)), Some((t1, p1))) = (before, after) {
        let p = p0 + (p1 - p0) * (boundary - t0) / (t1 - t0);
        let toward_inside = if is_left { t1 - boundary } else { boundary - t0 };
        let toward_outside = if is_left { boundary - t0 } else { t1 - boundary };
        let window = (other - boundary).abs();
        let gap = toward_inside.min(toward_outside).min(window);
        return (EdgeFix::Interpolated, Some(p), gap, (p1 - p0).abs());
    }
    (EdgeFix::Uncovered, None, f64::NAN, f64::NAN)
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

/// What a measured run cost.
#[derive(Clone, Debug)]
pub struct Run {
    /// Wall time the workload took.
    pub seconds: f64,
    /// Seconds the trapezoid covered. Shorter than `seconds` when an edge was uncovered.
    pub covered_s: f64,
    /// Time-weighted mean power over the covered span, in the backend's scope.
    pub mean_watts: f64,
    /// Idle power the caller measured, in the same scope.
    pub idle_watts: f64,
    /// How much that baseline wandered, so a reader can see whether the delta cleared it.
    pub idle_sigma: f64,
    /// Integral over the covered span, idle included. An uncovered edge is not filled in.
    pub joules_total: f64,
    /// `joules_total − idle_watts × covered_s`.
    ///
    /// This is the number to attribute to the computation. Reported beside `joules_total` rather
    /// than instead of it, because the subtraction is a modelling choice and hiding it would make
    /// two different quantities look like one.
    pub joules_above_idle: f64,
    /// Power readings whose timestamps fell inside the closed window.
    pub samples: usize,
    /// Median sample spacing, in seconds.
    pub cadence_s: f64,
    /// How the start of the workload window was placed on the sample series.
    pub left_edge: EdgeFix,
    /// How the end of the workload window was placed on the sample series.
    pub right_edge: EdgeFix,
    /// Seconds at the two edges that were not integrated.
    pub uncovered_s: f64,
    /// Edge bound from [`PowerIntegral::cadence_edge_j`], in joules, over the window.
    pub cadence_edge_j: f64,
    /// What produced these numbers.
    pub machine: String,
    /// Which power backend produced these readings.
    pub backend: &'static str,
}

impl Meter {
    /// Open a backend, or `None` if this machine exposes none.
    ///
    /// `None` means **not found here**, never "impossible" — a Linux box with INA3221 rails has the
    /// same counters and no backend yet.
    pub fn detect() -> Option<Meter> {
        // All of them, in order. `ina3221` once shipped with its own tests while `detect` still
        // returned only macmon -- so on the one class of machine that backend exists for, the
        // normal entry point could not reach it. A backend nothing routes to is a backend nobody
        // runs, and every new one has to be added here in the same commit.
        //
        // RAPL before INA3221 because they cover different machines and RAPL is the cheaper probe:
        // it reads a file, where INA3221 walks a sysfs tree looking for labelled rails.
        Meter::macmon().or_else(Meter::rapl).or_else(Meter::ina3221)
    }

    /// The INA3221 backend: shunt monitors on the board, polled from sysfs.
    ///
    /// `None` when no INA3221 is present or readable. See [`ina3221`] for what is tested and what
    /// is not -- no reading from this path has come from real hardware.
    ///
    /// # Panics
    ///
    /// Never. A missing or unreadable monitor is `None`, not a panic.
    #[must_use]
    pub fn ina3221() -> Option<Meter> {
        let rails = ina3221::Rails::detect().ok()?;
        let machine = format!("INA3221 rail {}", rails.total().label);
        let readings: Arc<Mutex<Vec<(Instant, f64)>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&readings);
        std::thread::spawn(move || loop {
            let Ok(w) = rails.watts() else { return };
            {
                let mut v = sink.lock().unwrap();
                v.push((Instant::now(), w));
                if v.len() > 4096 {
                    v.drain(..2048);
                }
            }
            std::thread::sleep(ina3221::POLL);
        });
        // A board rail carries everything downstream of it, discrete parts included.
        let m = Meter { child: None, readings, machine, backend: "ina3221", scope: backend_scope("ina3221", "") };
        m.wait_for_first(Duration::from_secs(4))?;
        Some(m)
    }

    /// The x86 Linux backend: Intel RAPL energy counters via `/sys/class/powercap`.
    ///
    /// Unlike the other two this reads ENERGY rather than power, so a window's cost is a
    /// subtraction instead of an integral estimated from samples — which removes the "workload
    /// shorter than the sampling interval" failure the others have to refuse.
    ///
    /// See the `rapl` module for the domain that lies: `psys` is documented as whole-platform, reads 0.2 W
    /// on the machine this was written against whether idle or running twenty busy cores, and is
    /// rejected by checking it against the package it is supposed to contain.
    ///
    /// # Panics
    ///
    /// Never. An unreadable counter is `None`.
    #[must_use]
    pub fn rapl() -> Option<Meter> {
        let (domain, kind) = rapl::choose()?;
        let readings: Arc<Mutex<Vec<(Instant, f64)>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&readings);
        std::thread::spawn(move || {
            let path = domain.path.join("energy_uj");
            let read = |p: &std::path::Path| -> Option<u64> {
                std::fs::read_to_string(p).ok()?.trim().parse().ok()
            };
            let Some(mut prev) = read(&path) else { return };
            let mut last = Instant::now();
            loop {
                std::thread::sleep(Duration::from_millis(INTERVAL_MS));
                let Some(now) = read(&path) else { return };
                let dt = last.elapsed().as_secs_f64();
                if dt > 0.0 {
                    let w = domain.delta_j(prev, now) / dt;
                    let mut v = sink.lock().unwrap();
                    v.push((Instant::now(), w));
                    if v.len() > 4096 {
                        v.drain(..2048);
                    }
                }
                prev = now;
                last = Instant::now();
            }
        });
        let cpu = std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("model name"))
                    .and_then(|l| l.split(':').nth(1))
                    .map(|v| v.trim().to_string())
            })
            .unwrap_or_else(|| "unknown x86".into());
        let machine = format!("{cpu} [{}]", rapl::scope_note(kind));
        let scope = backend_scope("rapl", kind);
        let m = Meter { child: None, readings, machine, backend: "rapl", scope };
        m.wait_for_first(Duration::from_secs(3))?;
        Some(m)
    }

    /// The macOS backend: `macmon pipe`, reading the `SoC`'s own counters.
    ///
    /// # Panics
    ///
    /// Never. A missing `macmon` is `None`.
    #[must_use]
    pub fn macmon() -> Option<Meter> {
        let mut child = Command::new("macmon")
            .args(["pipe", "-s", "0", "-i", &INTERVAL_MS.to_string()])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        let out = child.stdout.take()?;
        let readings: Arc<Mutex<Vec<(Instant, f64)>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&readings);
        std::thread::spawn(move || {
            for line in BufReader::new(out).lines().map_while(Result::ok) {
                if let Some(w) = field(&line, "sys_power") {
                    let mut v = sink.lock().unwrap();
                    v.push((Instant::now(), w));
                    // A long-lived meter would otherwise grow without bound. Keeping the last few
                    // minutes is far more than any window needs.
                    if v.len() > 4096 {
                        v.drain(..2048);
                    }
                }
            }
        });
        let machine = Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "unknown Apple silicon".into());
        // `sys_power` is SoC telemetry, integrated GPU included. It is not a wall-plug reading,
        // and a discrete GPU on its own feed is outside it.
        let m = Meter { child: Some(child), readings, machine, backend: "macmon", scope: backend_scope("macmon", "") };
        // macmon does not produce its first reading for over a second. Waiting for it here means a
        // caller's measurement window covers their workload rather than the backend's warm-up --
        // which is what made a 1.2 s idle window collect exactly one sample.
        m.wait_for_first(Duration::from_secs(4))?;
        Some(m)
    }

    /// Block until the backend has produced a reading, so the caller's first window is real.
    fn wait_for_first(&self, up_to: Duration) -> Option<()> {
        let deadline = Instant::now() + up_to;
        while Instant::now() < deadline {
            if !self.readings.lock().unwrap().is_empty() {
                return Some(());
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        None
    }

    /// Readings as `(seconds since origin, watts)`. A sample from before `origin` is negative.
    fn readings_secs(&self, origin: Instant) -> Vec<(f64, f64)> {
        self.readings
            .lock()
            .unwrap()
            .iter()
            .map(|(t, w)| {
                let dt = if *t >= origin {
                    t.duration_since(origin).as_secs_f64()
                } else {
                    -(origin.duration_since(*t).as_secs_f64())
                };
                (dt, *w)
            })
            .collect()
    }

    /// What this is measuring.
    /// What this backend's readings cover. Check it before attributing a run to a device.
    #[must_use]
    pub fn scope(&self) -> Scope {
        self.scope
    }

    #[must_use]
    /// The machine this meter is reading, as the backend names it.
    pub fn machine(&self) -> &str {
        &self.machine
    }

    #[must_use]
    /// Which backend is open: `macmon`, `rapl`, or `ina3221`.
    ///
    /// The harness uses this to decide whether a trial is a power integral (`macmon`,
    /// `ina3221`) or a counter delta (`rapl`). Those are different measurements.
    pub fn backend(&self) -> &'static str {
        self.backend
    }

    /// Mean power with nothing running, over `at_least`.
    ///
    /// Take it immediately before or after the run rather than once at startup: idle drifts with
    /// temperature and with whatever else the machine is doing, and a stale baseline subtracts the
    /// wrong number.
    ///
    /// # Errors
    ///
    /// A message when too few readings arrived for the window, or the machine is not idle -- a busy
    /// machine has no idle, and a baseline taken on one makes every later delta read low or negative.
    ///
    /// # Panics
    ///
    /// Never: a busy machine or a short window is an `Err`, not a panic.
    pub fn idle(&mut self, at_least: Duration) -> Result<Baseline, String> {
        let baseline = self.power_window(at_least)?;
        // A BUSY MACHINE HAS NO IDLE. Checked here rather than trusted, because the failure is
        // silent and it has already cost this project a published number: an earlier 86 ns/flip
        // figure was contaminated by concurrent background load and had to be corrected. Worse,
        // the contamination biases in the dangerous direction -- other people's work inflates the
        // baseline, so idle looks LARGER than it is, and any argument that turns on how much of
        // the bill is idle comes out overstated. Measured here at load 82 with five other sessions
        // running experiments, this backend reported a 67.6 W "idle" on a machine whose real idle
        // is a fraction of that.
        //
        // This is the quiet-host gate. It is not optional on this path, and the threshold is
        // [`ferrotherm::host::QUIET_LOAD`]. A contended reading uses [`Self::observe_idle`] and
        // must be labelled as contended by the caller. This function does not do that labelling,
        // and it does not accept the load.
        quiet_enough(load_average())?;
        Ok(baseline)
    }

    /// The same window as [`Self::idle`], without the quiet-host gate.
    ///
    /// The number is a baseline taken while other work may be running. It is not a quiet baseline.
    /// Callers that use it have to grade the result `contended host` and must not present it as quiet.
    ///
    /// # Errors
    ///
    /// A message when too few readings arrived. Load is recorded on the baseline and is not an error.
    ///
    /// # Panics
    ///
    /// Never.
    pub fn observe_idle(&mut self, at_least: Duration) -> Result<Baseline, String> {
        self.power_window(at_least)
    }

    fn power_window(&mut self, at_least: Duration) -> Result<Baseline, String> {
        // Settle first. Measured without this on a machine that had just finished a build: idle
        // came out at 67.6 W and the workload that followed averaged 65.9 W, so the subtraction
        // went negative and the "measurement" reported that computing costs nothing. A baseline is
        // only a baseline once the machine has stopped doing the last thing.
        std::thread::sleep(SETTLE);
        let from = Instant::now();
        std::thread::sleep(at_least.max(Duration::from_millis(MIN_SAMPLES as u64 * INTERVAL_MS)));
        let seconds = from.elapsed().as_secs_f64();
        let series = self.readings_secs(from);
        let n = series.iter().filter(|(t, _)| *t >= -EDGE_EPS && *t <= seconds + EDGE_EPS).count();
        if n < MIN_SAMPLES {
            return Err(format!(
                "idle needs at least {MIN_SAMPLES} readings and got {n} in {:.2} s; the backend \
                 ticks about every {INTERVAL_MS} ms, so ask for a longer window",
                at_least.as_secs_f64()
            ));
        }
        let integ = trapezoid(&series, 0.0, seconds)?;
        if integ.covered_s <= 0.0 {
            return Err("the idle window produced no covered span".into());
        }
        let watts = integ.joules / integ.covered_s;
        let inside: Vec<f64> = series
            .iter()
            .filter(|(t, _)| *t >= -EDGE_EPS && *t <= seconds + EDGE_EPS)
            .map(|(_, w)| *w)
            .collect();
        let amean = inside.iter().sum::<f64>() / inside.len() as f64;
        let var = inside.iter().map(|w| (w - amean).powi(2)).sum::<f64>() / inside.len() as f64;
        Ok(Baseline { watts, sigma: var.sqrt(), samples: integ.samples, load1: load_average(), cadence_s: integ.cadence_s })
    }

    /// Integrate power over `f` and return both gross and above-idle joules, including a delta
    /// inside the noise. [`Self::measure`] still refuses that delta. This one returns it so a
    /// contended run can show the number under its own grade instead of dropping it.
    ///
    /// # Errors
    ///
    /// Too few samples for an integral, or the edges leave fewer than two knots.
    pub fn integrate<R>(&mut self, idle: Baseline, f: impl FnOnce() -> R) -> Result<(Run, R), String> {
        let t0 = Instant::now();
        let value = f();
        let seconds = t0.elapsed().as_secs_f64();
        std::thread::sleep(Duration::from_millis(INTERVAL_MS + INTERVAL_MS / 2));
        Ok((self.integrate_window(idle, t0, seconds)?, value))
    }

    /// Run `f`, sampling the backend's scope throughout.
    ///
    /// # Errors
    ///
    /// A message when the workload drew less above the baseline than the baseline's own drift, since
    /// dividing that delta by a work count reports precision the measurement does not have.
    pub fn measure<R>(&mut self, idle: Baseline, f: impl FnOnce() -> R) -> Result<Run, String> {
        let t0 = Instant::now();
        let _ = f();
        let seconds = t0.elapsed().as_secs_f64();

        // The backend ticks on its own clock, so the reading covering the end of the window has not
        // necessarily arrived. Wait one interval for it rather than truncating the run.
        std::thread::sleep(Duration::from_millis(INTERVAL_MS + INTERVAL_MS / 2));
        let run = self.integrate_window(idle, t0, seconds)?;
        let mean = run.mean_watts;
        let idle_watts = idle.watts;
        let samples = run.samples;

        // A run that drew LESS than its baseline did not generate power; the baseline was taken
        // while the machine was still busy. Clamping this to zero was the first version, and it
        // turned a broken measurement into the confident claim that computing is free -- which is
        // exactly the failure this crate exists to prevent, committed by the crate itself.
        // A delta smaller than the baseline's own wander is noise wearing a decimal point. Three
        // sigma, and the message says what to do about it -- run a bigger model, not more repeats,
        // because repeating a measurement that cannot resolve the signal averages noise.
        // NOT `delta > 0.0 && delta < floor`. That was the first version and it left a hole a real
        // run fell into: a delta of -0.05 W is neither below the 0.98 busy-machine threshold nor
        // inside the `delta > 0` branch, so it passed both guards and `max(0.0)` turned it into
        // zero joules -- the confident claim that the computation was free, arrived at by two
        // guards each deciding it was the other one's problem.
        let delta = mean - idle_watts;
        let floor = (3.0 * idle.sigma).max(0.5);
        if delta < floor {
            return Err(format!(
                "the run drew {delta:.2} W above a baseline that wanders by {:.2} W (1 sigma over \
                 {} readings). That delta is inside the noise -- at or below zero it is not even \
                 the right sign -- so dividing it by the node count would report precision this \
                 measurement does not have. Use a larger model so the workload draws at least \
                 {floor:.1} W above idle.",
                idle.sigma, idle.samples
            ));
        }

        if mean < idle_watts * 0.98 {
            return Err(format!(
                "the run averaged {mean:.1} W against a {idle_watts:.1} W baseline. A computation \
                 does not draw less than idle, so the baseline was taken while the machine was \
                 still busy. Measure idle again with the machine quiet."
            ));
        }

        if samples < MIN_SAMPLES {
            return Err(format!(
                "that workload ran for {seconds:.3} s and produced {samples} power reading(s). \
                 The backend ticks about every {INTERVAL_MS} ms and a mean of fewer than \
                 {MIN_SAMPLES} readings is not an estimate. Run more sweeps, or a larger model, so \
                 the window is at least {:.1} s.",
                MIN_SAMPLES as f64 * INTERVAL_MS as f64 / 1000.0
            ));
        }

        // No clamp: a delta below the noise floor is refused above, so anything reaching here
        // has cleared it. A `.max(0.0)` here would be a second chance for a bad measurement to
        // become a plausible zero. The joules are already the trapezoid over the covered span.
        Ok(run)
    }

    fn integrate_window(&self, idle: Baseline, t0: Instant, seconds: f64) -> Result<Run, String> {
        let series = self.readings_secs(t0);
        let samples = series.iter().filter(|(t, _)| *t >= -EDGE_EPS && *t <= seconds + EDGE_EPS).count();
        if samples < MIN_SAMPLES {
            return Err(format!(
                "that workload ran for {seconds:.3} s and produced {samples} power reading(s). \
                 The backend ticks about every {INTERVAL_MS} ms and a mean of fewer than \
                 {MIN_SAMPLES} readings is not an estimate.",
            ));
        }
        let integ = trapezoid(&series, 0.0, seconds)?;
        if integ.covered_s <= 0.0 {
            return Err("the window produced no covered span".into());
        }
        let mean = integ.joules / integ.covered_s;
        let delta = mean - idle.watts;
        Ok(Run {
            seconds,
            covered_s: integ.covered_s,
            mean_watts: mean,
            idle_watts: idle.watts,
            idle_sigma: idle.sigma,
            joules_total: integ.joules,
            joules_above_idle: delta * integ.covered_s,
            samples: integ.samples,
            cadence_s: integ.cadence_s,
            left_edge: integ.left,
            right_edge: integ.right,
            uncovered_s: integ.uncovered_left_s + integ.uncovered_right_s,
            cadence_edge_j: integ.cadence_edge_j(),
            machine: self.machine.clone(),
            backend: self.backend,
        })
    }
}

impl Drop for Meter {
    fn drop(&mut self) {
        if let Some(c) = self.child.as_mut() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

impl Run {
    /// Per-operation prices derived from this run's counts.
    ///
    /// **What this can and cannot separate.** The ledger counts samples, reads and writes; one
    /// measurement gives one number. Three unknowns from one equation is not a calibration, so this
    /// attributes everything to the operation that dominated and refuses when none did — running a
    /// sample-heavy workload and a write-heavy one and solving the pair is the real procedure, and
    /// this is the honest single-workload version of it.
    ///
    /// # Errors
    ///
    /// A message when the run counted no operations, or was not sample-dominated. One measurement
    /// cannot separate three costs; the honest procedure is a sample-heavy run and a write-heavy one.
    pub fn prices_from(&self, l: &Ledger) -> Result<Prices, String> {
        let (s, r, w) = (l.samples as f64, l.reads as f64, l.writes as f64);
        let total = s + r + w;
        if total == 0.0 {
            return Err("that run counted no operations, so there is nothing to divide by".into());
        }
        if s / total < 0.99 {
            return Err(format!(
                "this derives a per-SAMPLE price and needs a sample-dominated run; that one was \
                 {:.1}% samples, {:.1}% reads, {:.1}% writes. One measurement cannot separate \
                 three costs -- measure a sample-heavy run and a write-heavy run and solve the \
                 pair.",
                100.0 * s / total,
                100.0 * r / total,
                100.0 * w / total
            ));
        }
        Ok(Prices {
            e_sample: self.joules_above_idle / s,
            // Not zero: unmeasured. Zero would be a claim that reads and writes are free, which is
            // the exact shape of the error this crate exists to stop.
            e_read: f64::NAN,
            e_write: f64::NAN,
            reflash_hz_cap: None,
            // THIS crate is the one that meters. A price it derives comes from an instrument, an
            // idle baseline and a stated reading count, which is what `Metered` means -- and it is
            // the only place in the workspace entitled to construct one.
            evidence: ferrotherm::ledger::Evidence::Metered,
            // Leaked deliberately: `Prices::source` is &'static str, and a measurement's provenance
            // has to travel with it. One string per derived price, for the life of the process.
            source: Box::leak(
                format!(
                    "measured on {} via {} — {:.4} J above idle over {:.2} s at {:.1} W ({:.1} W \
                     idle), {} readings, divided by {} node updates. e_sample only; read and write \
                     are unmeasured, not zero.",
                    self.machine,
                    self.backend,
                    self.joules_above_idle,
                    self.seconds,
                    self.mean_watts,
                    self.idle_watts,
                    self.samples,
                    l.samples
                )
                .into_boxed_str(),
            ),
        })
    }
}

/// Is this load average quiet enough for the reading to be called an idle baseline?
///
/// Split out of [`Meter::idle`] so it can be tested WITHOUT arranging a busy machine. The first
/// version of the guard lived inline, and its test could only assert something real when the
/// machine happened to be loaded -- on a quiet machine it took the other branch and would have
/// passed just as happily with the guard deleted. A test named after a property is not a test of it.
///
/// `None` means the platform exposes no load average, which is treated as permission to proceed:
/// refusing to measure at all on such a platform is a worse failure than measuring unguarded, and
/// [`Baseline::load1`] records that it was unknown.
fn quiet_enough(load1: Option<f64>) -> Result<(), String> {
    match load1 {
        Some(l) if l > QUIET_LOAD => Err(format!(
            "the 1-minute load average is {l:.1}, so this machine is not idle, and what \
             follows would be somebody else's work charged to yours. A load average counts \
             RUNNABLE THREADS, so anything above {QUIET_LOAD:.0} means at least that many are \
             never sleeping. Wait for the machine to go quiet and take the baseline again. \
             Note which way this biases: a contaminated baseline is INFLATED, so idle looks \
             larger than it is, not smaller."
        )),
        _ => Ok(()),
    }
}

/// Pull one numeric field out of a JSON object line.
///
/// A hand-rolled scan rather than a parser, because one field does not justify a dependency in a
/// crate whose sibling's headline is zero dependencies. It matches `"name":` followed by a number,
/// which is what the backend emits and nothing else in the line looks like.
fn field(line: &str, name: &str) -> Option<f64> {
    let key = format!("\"{name}\":");
    let at = line.find(&key)? + key.len();
    let rest = &line[at..];
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e' || c == '+'))
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A price this crate derives is Metered, and it is the only grade it may claim.
    ///
    /// The evidence grade travels with the number or it is not evidence. `meter` is the producer
    /// of metered prices in this workspace; every other `Prices` in the tree is Simulated, Derived
    /// or Unstated, and a price that came off an instrument has to be distinguishable from one
    /// that came off a spreadsheet.
    #[test]
    fn a_derived_price_is_graded_metered() {
        let m = Run {
            seconds: 10.0,
            covered_s: 10.0,
            mean_watts: 4.05,
            idle_watts: 3.5,
            idle_sigma: 0.008,
            joules_total: 40.5,
            joules_above_idle: 0.55,
            samples: 100,
            cadence_s: 0.1,
            left_edge: EdgeFix::OnSample,
            right_edge: EdgeFix::OnSample,
            uncovered_s: 0.0,
            cadence_edge_j: 0.0,
            machine: "test".into(),
            backend: "test",
        };
        let l = Ledger { samples: 1_000_000, reads: 0, writes: 0 };
        let p = m.prices_from(&l).expect("a sample-dominated run");
        assert_eq!(p.evidence, ferrotherm::ledger::Evidence::Metered);
        assert!(p.source.contains("measured on"));
        // and a comparison against anything weaker drops to the weaker grade
        assert_eq!(
            ferrotherm::ledger::weaker(p.evidence, ferrotherm::ledger::Z1_SPICE.evidence),
            ferrotherm::ledger::Evidence::Simulated
        );
    }


    /// Measuring power means OWNING the machine.
    ///
    /// Cargo runs tests in parallel, so two of these ran at once and each measured the other's
    /// load: one passed alone and failed in the suite, which is the signature of a benchmark that
    /// does not control what else is running. A std-only lock serialises them; nothing here needs
    /// a test-ordering dependency to say something this simple.
    static MACHINE: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Report a skip, and refuse to skip when the caller has said every check must run.
    ///
    /// **A SKIPPING TEST PASSES**, which is indistinguishable from a passing one unless it is
    /// asked. Every other gate in this repository already has this lever —
    /// `check-semantics`, `check-answers` and the mutation suite all honour
    /// `FERROTHERM_REQUIRE_ALL` — and the meter tests did not, which is how
    /// `a_real_workload_measures_above_idle_and_derives_a_price` carried a WRONG assertion for as
    /// long as it did: on any machine with a load average above 2 it returned before reaching it,
    /// printed a line nobody reads, and reported ok.
    fn skipped(reason: &str) {
        if std::env::var_os("FERROTHERM_REQUIRE_ALL").is_some() {
            panic!("FERROTHERM_REQUIRE_ALL is set and this check did not run: {reason}");
        }
        eprintln!("{reason}; skipping");
    }

    macro_rules! meter_or_skip {
        () => {{
            // Held for the body of the test, so no other test perturbs the readings.
            let own = MACHINE.lock().unwrap_or_else(|e| e.into_inner());
            match Meter::detect() {
                Some(m) => (m, own),
                None => {
                    skipped("no power backend on this machine");
                    return;
                }
            }
        }};
    }

    #[test]
    fn the_field_scanner_reads_what_the_backend_emits() {
        let line = r#"{"all_power":16.46,"ane_power":0.0,"sys_power":50.25,"temp":{"a":1}}"#;
        assert_eq!(field(line, "sys_power"), Some(50.25));
        assert_eq!(field(line, "ane_power"), Some(0.0));
        assert_eq!(field(line, "all_power"), Some(16.46));
        assert_eq!(field(line, "not_here"), None);
        // A negative and an exponent, both of which appear in real output.
        assert_eq!(field(r#"{"x":-1.5e-3}"#, "x"), Some(-1.5e-3));
    }

    #[test]
    fn macmon_sys_power_is_soc_telemetry_and_does_not_cover_a_discrete_gpu() {
        assert_eq!(backend_scope("macmon", ""), Scope::SocTelemetry);
        assert_eq!(Scope::SocTelemetry.label(), "soc-telemetry");
        assert!(Scope::SocTelemetry.covers(false));
        assert!(!Scope::SocTelemetry.covers(true), "sys_power does not see a discrete card");
        assert_eq!(backend_scope("rapl", "package"), Scope::CpuPackage);
        assert_eq!(backend_scope("rapl", "psys"), Scope::WholeSystem);
        assert!(Scope::CpuPackage.covers(false));
        assert!(!Scope::CpuPackage.covers(true));
    }

    #[test]
    fn trapezoid_uses_the_timestamps_and_does_not_fill_an_uncovered_edge() {
        let flat = [(0.0, 10.0), (1.0, 10.0), (2.0, 10.0)];
        let even = trapezoid(&flat, 0.0, 2.0).unwrap();
        assert!((even.joules - 20.0).abs() < 1e-12);
        assert_eq!(even.samples, 3);
        assert!((even.cadence_s - 1.0).abs() < 1e-12);
        assert_eq!(even.left, EdgeFix::OnSample);
        assert_eq!(even.right, EdgeFix::OnSample);
        assert_eq!(even.uncovered_left_s + even.uncovered_right_s, 0.0);
        assert_eq!(even.cadence_edge_j(), 0.0);

        // Arithmetic mean of these samples is 50, and 50 W times 2 s is 100 J.
        // The trapezoid weights the short jump and is 95 J.
        let uneven = [(0.0, 0.0), (1.0, 0.0), (1.1, 100.0), (2.0, 100.0)];
        let trap = trapezoid(&uneven, 0.0, 2.0).unwrap();
        assert!((trap.joules - 95.0).abs() < 1e-9, "{}", trap.joules);
        let mean = uneven.iter().map(|(_, p)| p).sum::<f64>() / uneven.len() as f64;
        assert!((mean * 2.0 - 100.0).abs() < 1e-9);
        assert!((trap.joules - mean * 2.0).abs() > 1.0);

        let bridged = [(-0.2, 0.0), (0.2, 10.0), (0.8, 10.0), (1.2, 20.0)];
        let edges = trapezoid(&bridged, 0.0, 1.0).unwrap();
        assert!((edges.joules - 10.0).abs() < 1e-9, "{}", edges.joules);
        assert_eq!(edges.left, EdgeFix::Interpolated);
        assert_eq!(edges.right, EdgeFix::Interpolated);
        assert_eq!(edges.samples, 2);
        assert!((edges.cadence_s - 0.4).abs() < 1e-12);
        assert!((edges.cadence_edge_j() - 2.0).abs() < 1e-9, "{}", edges.cadence_edge_j());

        let open = [(0.25, 4.0), (0.75, 8.0)];
        let missing = trapezoid(&open, 0.0, 1.0).unwrap();
        assert!((missing.joules - 3.0).abs() < 1e-12, "{}", missing.joules);
        assert!((missing.covered_s - 0.5).abs() < 1e-12);
        assert!((missing.uncovered_left_s - 0.25).abs() < 1e-12);
        assert!((missing.uncovered_right_s - 0.25).abs() < 1e-12);
        assert_eq!(missing.left, EdgeFix::Uncovered);
        assert_eq!(missing.right, EdgeFix::Uncovered);
        assert!((missing.cadence_edge_j() - 2.0).abs() < 1e-12, "{}", missing.cadence_edge_j());
        assert_eq!(missing.samples, 2);
        assert!((missing.cadence_s - 0.5).abs() < 1e-12);
    }

    #[test]
    fn the_load_average_is_readable_on_this_platform() {
        // If this returns None the guard is inert here, which is a fact worth knowing rather than
        // a thing to discover from a contaminated publication months later.
        match load_average() {
            Some(l) => assert!(l >= 0.0 && l.is_finite(), "implausible load average: {l}"),
            None => eprintln!(
                "no load average on this platform; Meter::idle cannot check that the machine is \
                 quiet, and any baseline it takes is unguarded"
            ),
        }
    }

    #[test]
    fn the_quiet_threshold_refuses_a_loaded_machine_and_names_the_reason() {
        // Tested on the DECISION, not on whatever this machine happens to be doing, so deleting
        // the guard fails here on a quiet machine and a busy one alike.
        assert!(quiet_enough(Some(0.0)).is_ok());
        assert!(quiet_enough(Some(QUIET_LOAD)).is_ok(), "the threshold itself is allowed");
        assert!(
            quiet_enough(None).is_ok(),
            "a platform with no load average must still be measurable; the unknown is recorded"
        );

        // 82 is not arbitrary: it is the load this project actually published a table at.
        let e = quiet_enough(Some(82.0)).unwrap_err();
        assert!(e.contains("load average"), "the refusal must name the reason: {e}");
        assert!(e.contains("82"), "and the value it saw: {e}");
        assert!(
            e.contains("INFLATED"),
            "and which way the contamination biases, because that is what makes it dangerous: {e}"
        );
        // Just over the line is refused too, or the threshold is decorative.
        assert!(quiet_enough(Some(QUIET_LOAD + 0.01)).is_err());
    }

    #[test]
    fn idle_actually_consults_the_quiet_threshold() {
        // The unit test above proves the decision is right; this proves `idle` asks it. Whichever
        // side of the line this machine is on, the two must agree -- a guard that exists and is
        // never called is the failure mode the split was made to expose.
        let (mut m, _own) = meter_or_skip!();
        let observed = load_average();
        let verdict = m.idle(Duration::from_millis(1200));
        match (observed, &verdict) {
            (Some(l), Err(e)) if l > QUIET_LOAD => {
                assert!(e.contains("load average"), "refused for the wrong reason: {e}");
            }
            (Some(l), Ok(b)) => {
                assert!(l <= QUIET_LOAD, "granted a baseline at load {l}, above {QUIET_LOAD}");
                assert_eq!(b.load1.map(|x| x <= QUIET_LOAD), Some(true));
            }
            // No load average, or refused for one of the other reasons (too few readings). Both
            // are legitimate outcomes and neither says anything about this guard.
            _ => {}
        }
    }

    #[test]
    fn a_workload_too_short_to_measure_is_refused() {
        // The failure this prevents is a mean computed from one reading. The backend ticks on its
        // own clock, so a fast workload simply is not observable, and returning a number anyway
        // would be inventing precision.
        let (mut m, _own) = meter_or_skip!();
        let zero = Baseline { watts: 0.0, sigma: 0.0, samples: MIN_SAMPLES, load1: None, cadence_s: 0.1 };
        let e = m.measure(zero, || std::hint::black_box(1 + 1)).unwrap_err();
        assert!(e.contains("reading"), "must say what was short: {e}");
        assert!(e.contains("not an estimate"), "and why that is refused: {e}");
    }

    #[test]
    fn a_real_workload_measures_above_idle_and_derives_a_price() {
        use ferrotherm::{gibbs::Sampler, ising::lattice2d, ledger::Ledger};
        let (mut m, _own) = meter_or_skip!();
        let g = lattice2d(256, 1.0);

        // SIZE THE WORKLOAD FROM THIS MACHINE, not from a number written down about another one.
        //
        // This test used to run a fixed 2,000 sweeps, with a comment reading "at roughly 60M node
        // updates a second this is about two seconds of work". The machine now does about 200M, so
        // the window fell to 0.662 s and produced 7 readings against a floor of 8 — and the meter's
        // own guard refused, correctly, to call that an estimate. The fixture encoded an assumption
        // about the hardware and went stale when the hardware did not.
        //
        // Timed BEFORE the idle baseline on purpose: this warms the machine, and the baseline that
        // follows then measures a settled one rather than a cooling one.
        let probe = 200usize;
        let t0 = std::time::Instant::now();
        {
            let mut s = Sampler::new(&g, 0.7, 1);
            s.sweeps(probe, None);
        }
        let per_sweep = t0.elapsed().as_secs_f64() / probe as f64;
        // Two seconds of work, which is comfortably above the 0.8 s the readings guard demands and
        // leaves room for the machine to be faster on the real run than on the probe.
        let sweeps = ((2.0 / per_sweep).ceil() as usize).clamp(probe, 200_000);
        // The same treatment the `measure` call below already had, applied to the line that
        // panicked instead: a contaminated BASELINE is as much "your laptop was compiling" as a
        // contaminated run, and this `.unwrap()` is what went red while a concurrent clippy build
        // was running. The refusal firing is the meter working, not the code being wrong.
        let idle = match m.idle(Duration::from_millis(1200)) {
            Ok(i) => i,
            Err(e) => {
                skipped(&format!("machine was not quiet enough to take a baseline: {e}"));
                return;
            }
        };
        assert!(idle.watts > 0.0, "a running machine draws power");

        let mut led = Ledger::default();
        // A machine that is busy with something else cannot be measured, and the library says so
        // rather than returning a number. Treat that like "no backend": skip, do not fail. This
        // test really did go red once because the build had not finished settling, and a suite that
        // turns red for "your laptop was compiling" is a suite people learn to ignore.
        let run = match m.measure(idle, || {
            let mut s = Sampler::new(&g, 0.7, 1);
            s.sweeps(sweeps, Some(&mut led));
        }) {
            Ok(r) => r,
            Err(e) if e.contains("still busy") || e.contains("inside the noise") => {
                skipped(&format!("machine was not quiet enough to measure: {e}"));
                return;
            }
            Err(e) => panic!("{e}"),
        };
        // Not a seconds assertion: how long 131M updates take depends on what else the machine is
        // doing, and this went red at 0.732 s only because the box had got quieter. The guard that
        // matters is the sample count, checked below.

        assert!(run.samples >= MIN_SAMPLES, "{run:?}");
        assert!(run.mean_watts > 0.0 && run.seconds > 0.0, "{run:?}");
        assert!(
            run.joules_total >= run.joules_above_idle,
            "the total includes idle, so it cannot be the smaller number: {run:?}"
        );

        let p = run.prices_from(&led).unwrap();
        assert!(p.e_sample > 0.0 && p.e_sample.is_finite(), "e_sample = {}", p.e_sample);
        assert!(p.source.contains("measured on"), "provenance must travel: {}", p.source);
        assert!(
            p.e_read.is_nan() && p.e_write.is_nan(),
            "reads and writes were not measured here, and zero would claim they are free"
        );
        // NOT stated as a whole table: one measurement fixes one of three costs.
        assert!(!p.is_stated(), "one measured term does not make a complete price table");

        // AND YET IT PRICES THIS RUN EXACTLY, which is the whole point of `Ledger::joules` charging
        // PER OPERATION rather than all-or-nothing. A serial sweep performs samples and nothing
        // else, so the two unstated costs are multiplied by zero and the answer is real.
        //
        // This assertion used to read `assert_eq!(led.joules(&p), None)`. That was correct under the
        // OLD all-or-nothing rule and became wrong the day `joules` changed — the change whose own
        // documentation says "under the old rule that measurement could price NOTHING, including
        // the very workload it was taken from". The test contradicted the behaviour it exists to
        // check, and survived because it returns early on any machine with a load average above 2:
        // it reached this line perhaps once in a preflight run. See `skipped`.
        assert_eq!(led.reads, 0, "a serial sweep reads nothing out");
        assert_eq!(led.writes, 0, "and flashes nothing");
        let total = led.joules(&p).expect("a sample-only ledger is priceable by a sample-only table");
        assert!(
            (total - led.samples as f64 * p.e_sample).abs() < 1e-9 * total.abs().max(1.0),
            "the total is the sampling cost and nothing else: {total} against {} samples at {}",
            led.samples,
            p.e_sample
        );
        // And the refusal still refuses where it should: an operation the run DID perform, with no
        // price for it, yields None rather than a number.
        let with_reads = Ledger { reads: 1, ..led };
        assert_eq!(
            with_reads.joules(&p),
            None,
            "one unpriced read must sink the total, or the refusal means nothing"
        );
    }

    #[test]
    fn a_mixed_workload_will_not_be_turned_into_a_per_sample_price() {
        // Three unknowns, one equation. Refusing is the whole point: the alternative is dividing
        // total energy by node updates and calling it e_sample while writes paid for most of it.
        let run = Run {
            seconds: 1.0,
            covered_s: 1.0,
            mean_watts: 30.0,
            idle_watts: 20.0,
            idle_sigma: 0.1,
            joules_total: 30.0,
            joules_above_idle: 10.0,
            samples: 10,
            cadence_s: 0.1,
            left_edge: EdgeFix::OnSample,
            right_edge: EdgeFix::OnSample,
            uncovered_s: 0.0,
            cadence_edge_j: 0.0,
            machine: "test".into(),
            backend: "test",
        };
        let mixed = Ledger { samples: 100, reads: 50, writes: 10 };
        let e = run.prices_from(&mixed).unwrap_err();
        assert!(e.contains("sample-dominated"), "{e}");
        assert!(e.contains("solve the pair"), "and say how to do it properly: {e}");

        let clean = Ledger { samples: 10_000, reads: 0, writes: 0 };
        assert!(run.prices_from(&clean).is_ok());
    }
}
