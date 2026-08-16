//! Measured wall power, so a joules figure can describe the machine that produced it.
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
//! Whole-system wall power, integrated over the run. Then **idle is subtracted**: a machine that
//! draws 20 W doing nothing does not charge that to your workload, and a "measurement" that
//! forgets to subtract it reports mostly the cost of the computer being switched on. Both numbers
//! are reported so the subtraction is visible rather than assumed.
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
//! - **macOS**: `macmon pipe`, which reads the SoC's own power counters. Detected on `PATH`.
//! - **Jetson / Linux**: the INA3221 rails under `/sys/bus/i2c/.../in_power*_input` are the
//!   equivalent. Not implemented here — the Jetson on this tailnet has been offline for a week, and
//!   a backend nobody can run is a backend nobody has tested. The trait is the seam it slots into.
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

/// A source of wall-power readings, in watts.
pub struct Meter {
    child: Child,
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
}

/// A measured idle baseline: its level AND its spread.
///
/// The spread is not decoration. Measured on an M5 Max sitting at ~60 W, two runs of the same
/// workload reported 2.34e-8 and 1.81e-7 J per node update -- an eight-fold spread -- because one
/// of them added about 1 W to a baseline that wanders by more than that on its own. A delta smaller
/// than the baseline's own variation is not a small measurement, it is not a measurement.
#[derive(Clone, Copy, Debug)]
pub struct Baseline {
    pub watts: f64,
    /// Standard deviation of the readings the baseline was computed from.
    pub sigma: f64,
    pub samples: usize,
}

/// What a measured run cost.
#[derive(Clone, Debug)]
pub struct Run {
    /// Wall time the workload took.
    pub seconds: f64,
    /// Mean whole-system power while it ran.
    pub mean_watts: f64,
    /// Mean whole-system power with the machine idle, measured separately.
    pub idle_watts: f64,
    /// How much that baseline wandered, so a reader can see whether the delta cleared it.
    pub idle_sigma: f64,
    /// Everything the wall paid during the run, idle included.
    pub joules_total: f64,
    /// What the workload added: `(mean − idle) × seconds`.
    ///
    /// This is the number to attribute to the computation. Reported beside `joules_total` rather
    /// than instead of it, because the subtraction is a modelling choice and hiding it would make
    /// two different quantities look like one.
    pub joules_above_idle: f64,
    /// Power readings collected inside the window.
    pub samples: usize,
    /// What produced these numbers.
    pub machine: String,
    pub backend: &'static str,
}

impl Meter {
    /// Open a backend, or `None` if this machine exposes none.
    ///
    /// `None` means **not found here**, never "impossible" — a Linux box with INA3221 rails has the
    /// same counters and no backend yet.
    pub fn detect() -> Option<Meter> {
        Meter::macmon()
    }

    /// The macOS backend: `macmon pipe`, reading the SoC's own counters.
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
        let m = Meter { child, readings, machine, backend: "macmon" };
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

    /// Mean of the readings that arrived between `from` and now.
    fn mean_since(&self, from: Instant) -> (f64, usize) {
        let v = self.readings.lock().unwrap();
        let inside: Vec<f64> = v.iter().filter(|(t, _)| *t >= from).map(|(_, w)| *w).collect();
        if inside.is_empty() {
            return (0.0, 0);
        }
        (inside.iter().sum::<f64>() / inside.len() as f64, inside.len())
    }

    /// What this is measuring.
    pub fn machine(&self) -> &str {
        &self.machine
    }

    /// Mean power with nothing running, over `at_least`.
    ///
    /// Take it immediately before or after the run rather than once at startup: idle drifts with
    /// temperature and with whatever else the machine is doing, and a stale baseline subtracts the
    /// wrong number.
    pub fn idle(&mut self, at_least: Duration) -> Result<Baseline, String> {
        // Settle first. Measured without this on a machine that had just finished a build: idle
        // came out at 67.6 W and the workload that followed averaged 65.9 W, so the subtraction
        // went negative and the "measurement" reported that computing costs nothing. A baseline is
        // only a baseline once the machine has stopped doing the last thing.
        std::thread::sleep(SETTLE);
        let from = Instant::now();
        std::thread::sleep(at_least.max(Duration::from_millis(MIN_SAMPLES as u64 * INTERVAL_MS)));
        let (mean, n) = self.mean_since(from);
        if n < MIN_SAMPLES {
            return Err(format!(
                "idle needs at least {MIN_SAMPLES} readings and got {n} in {:.2} s; the backend \
                 ticks about every {INTERVAL_MS} ms, so ask for a longer window",
                at_least.as_secs_f64()
            ));
        }
        let v = self.readings.lock().unwrap();
        let inside: Vec<f64> = v.iter().filter(|(t, _)| *t >= from).map(|(_, w)| *w).collect();
        let var = inside.iter().map(|w| (w - mean).powi(2)).sum::<f64>() / inside.len() as f64;
        Ok(Baseline { watts: mean, sigma: var.sqrt(), samples: n })
    }

    /// Run `f`, sampling wall power throughout.
    pub fn measure<R>(&mut self, idle: Baseline, f: impl FnOnce() -> R) -> Result<Run, String> {
        let idle_watts = idle.watts;
        let t0 = Instant::now();
        let _ = f();
        let seconds = t0.elapsed().as_secs_f64();

        // The backend ticks on its own clock, so the reading covering the end of the window has not
        // necessarily arrived. Wait one interval for it rather than truncating the run.
        std::thread::sleep(Duration::from_millis(INTERVAL_MS + INTERVAL_MS / 2));
        let (mean, samples) = self.mean_since(t0);

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

        Ok(Run {
            seconds,
            mean_watts: mean,
            idle_watts,
            idle_sigma: idle.sigma,
            joules_total: mean * seconds,
            // No clamp: a delta below the noise floor is refused above, so anything reaching here
            // has cleared it. A `.max(0.0)` here would be a second chance for a bad measurement to
            // become a plausible zero.
            joules_above_idle: delta * seconds,
            samples,
            machine: self.machine.clone(),
            backend: self.backend,
        })
    }
}

impl Drop for Meter {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
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

    /// Measuring power means OWNING the machine.
    ///
    /// Cargo runs tests in parallel, so two of these ran at once and each measured the other's
    /// load: one passed alone and failed in the suite, which is the signature of a benchmark that
    /// does not control what else is running. A std-only lock serialises them; nothing here needs
    /// a test-ordering dependency to say something this simple.
    static MACHINE: std::sync::Mutex<()> = std::sync::Mutex::new(());

    macro_rules! meter_or_skip {
        () => {{
            // Held for the body of the test, so no other test perturbs the readings.
            let own = MACHINE.lock().unwrap_or_else(|e| e.into_inner());
            match Meter::detect() {
                Some(m) => (m, own),
                None => {
                    eprintln!("no power backend on this machine; skipping");
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
    fn a_workload_too_short_to_measure_is_refused() {
        // The failure this prevents is a mean computed from one reading. The backend ticks on its
        // own clock, so a fast workload simply is not observable, and returning a number anyway
        // would be inventing precision.
        let (mut m, _own) = meter_or_skip!();
        let zero = Baseline { watts: 0.0, sigma: 0.0, samples: MIN_SAMPLES };
        let e = m.measure(zero, || std::hint::black_box(1 + 1)).unwrap_err();
        assert!(e.contains("reading"), "must say what was short: {e}");
        assert!(e.contains("not an estimate"), "and why that is refused: {e}");
    }

    #[test]
    fn a_real_workload_measures_above_idle_and_derives_a_price() {
        use ferrotherm::{gibbs::Sampler, ising::lattice2d, ledger::Ledger};
        let (mut m, _own) = meter_or_skip!();
        let idle = m.idle(Duration::from_millis(1200)).unwrap();
        assert!(idle.watts > 0.0, "a running machine draws power");

        // Big enough to be measurable, and sized from what the guard demands rather than guessed:
        // the first attempt ran for 59 ms and collected 2 readings, which the refusal caught. At
        // roughly 60M node updates a second this is about two seconds of work.
        let g = lattice2d(256, 1.0);
        let mut led = Ledger::default();
        // A machine that is busy with something else cannot be measured, and the library says so
        // rather than returning a number. Treat that like "no backend": skip, do not fail. This
        // test really did go red once because the build had not finished settling, and a suite that
        // turns red for "your laptop was compiling" is a suite people learn to ignore.
        let run = match m.measure(idle, || {
            let mut s = Sampler::new(&g, 0.7, 1);
            s.sweeps(2_000, Some(&mut led));
        }) {
            Ok(r) => r,
            Err(e) if e.contains("still busy") || e.contains("inside the noise") => {
                eprintln!("machine was not quiet enough to measure; skipping: {e}");
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
        // And therefore NOT stated as a whole table, so `joules` refuses to total a run with it.
        // That is the correct outcome rather than a shortfall: one measurement fixes one of three
        // costs, and a table that answered anyway would be inventing the other two.
        assert!(!p.is_stated(), "one measured term does not make a complete price table");
        assert_eq!(led.joules(&p), None);
    }

    #[test]
    fn a_mixed_workload_will_not_be_turned_into_a_per_sample_price() {
        // Three unknowns, one equation. Refusing is the whole point: the alternative is dividing
        // total energy by node updates and calling it e_sample while writes paid for most of it.
        let run = Run {
            seconds: 1.0,
            mean_watts: 30.0,
            idle_watts: 20.0,
            idle_sigma: 0.1,
            joules_total: 30.0,
            joules_above_idle: 10.0,
            samples: 10,
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
