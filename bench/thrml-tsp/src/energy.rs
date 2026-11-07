//! Measured joules, or an explicit absence.
//!
//! The instrument is a hardware counter: Intel RAPL through `ferrotherm_meter` on Linux,
//! the NVIDIA Management Library energy counter when `libnvidia-ml.so.1` loads, and on
//! macOS only the rootless `macmon` path already implemented in `ferrotherm_meter`.
//! This file does not spawn a privileged helper and does not read a vendor price table.
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
    /// Whatever `Meter::detect` found: macmon, RAPL, or INA3221. `None` is "not on this machine".
    pub wall: Option<Meter>,
    pub wall_note: String,
    pub nvml: Option<Nvml>,
    pub nvml_note: String,
    pub macos_policy: &'static str,
}

impl Probe {
    pub fn detect() -> Probe {
        let load1 = load_average();
        let quiet = load1.is_some_and(|l| l <= QUIET_LOAD);
        let wall = Meter::detect();
        let wall_note = match &wall {
            Some(m) => format!("machine={} scope={:?}", m.machine(), m.scope()),
            None => "no macmon, no RAPL powercap, no INA3221 rail on this machine".into(),
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
            nvml,
            nvml_note,
            macos_policy: "macOS power, when that OS is the host, is ferrotherm_meter::Meter::macmon \
                           (rootless IOReport via the macmon binary). This harness has no other macOS path.",
        }
    }
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

/// Take an idle baseline if a RAPL (or macmon, or INA3221) meter is open and the machine is quiet.
/// The meter's own `idle` refuses a busy machine. That refusal is the result, not a zero baseline.
pub fn idle_baseline(probe: &mut Probe) -> Result<ferrotherm_meter::Baseline, String> {
    let Some(meter) = probe.wall.as_mut() else {
        return Err("no wall-power backend".into());
    };
    if !probe.quiet {
        return Err(format!(
            "load1 {:?} is above {QUIET_LOAD}; a busy machine has no idle baseline",
            probe.load1
        ));
    }
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
    fn a_counter_wrap_is_not_negative_energy() {
        // 100 mJ, then a wrap to 50 mJ: 2^64 - 100 + 50 mJ, reported in joules.
        let d = mj_delta_j(100, 50);
        assert!(d > 0.0);
        assert!((mj_delta_j(0, 2500) - 2.5).abs() < 1e-12);
    }
}
