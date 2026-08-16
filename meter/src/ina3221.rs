//! INA3221 shunt monitors, as found on Jetson carrier boards and plenty of other Linux hardware.
//!
//! The macOS backend asks the SoC what it drew. This asks a current-sense chip on the board, which
//! is the same question posed to different silicon, and on a Jetson it is the only way to ask.
//!
//! # The rails do not add up, and that is the whole difficulty
//!
//! An INA3221 has three channels, and on a Jetson they are nested rather than disjoint: `VDD_IN`
//! is the **whole board's input**, and `VDD_CPU_GPU_CV` and `VDD_SOC` are parts of what `VDD_IN`
//! already counts. Summing the three channels — the obvious thing, and what a backend written from
//! the attribute names alone would do — roughly doubles the answer, and does it silently, producing
//! a number that looks entirely reasonable and is wrong on every measurement.
//!
//! So this reads [`in[123]_label`] and uses **one** rail, the total. When the labels do not say
//! which rail is total, it refuses and lists what it found, because there is no safe guess: picking
//! the largest would be a heuristic that fails exactly when a subsidiary rail spikes, and summing
//! would be wrong always.
//!
//! # Two driver layouts
//!
//! | driver | path | attributes | units |
//! |---|---|---|---|
//! | upstream `hwmon` | `.../hwmon/hwmonN/` | `in[123]_input` × `curr[123]_input` | mV × mA |
//! | L4T downstream `ina3221x` | `.../iio:deviceN/` | `in_power[012]_input` | mW |
//!
//! The upstream driver exposes **no power attribute at all** — bus voltage in mV and current in mA,
//! so power is `mV × mA / 1e6` watts. The downstream Jetson driver reports milliwatts directly.
//! Both are handled; which one a given board presents depends on its JetPack version.
//!
//! # What is verified, and what is not
//!
//! Everything here except the existence of a Jetson: rail discovery, label matching, the refusal
//! when no total rail can be identified, both unit conversions, and the arithmetic, are tested
//! against fixture directories built by the tests themselves — [`Rails::at`] takes a path precisely
//! so that is possible.
//!
//! **No reading in this module has been taken from real hardware.** The Jetson on our tailnet has
//! been offline, and this crate's own doctrine is that a backend nobody can run is a backend nobody
//! has tested. Calling the arithmetic verified and the hardware path unverified is the honest split,
//! and it is a better position than leaving the file empty — the part that was going to be wrong is
//! the part that is now tested.

use std::path::{Path, PathBuf};
use std::time::Duration;

/// Names a Jetson gives the rail that carries the whole board.
///
/// Matched case-insensitively against `in[123]_label`. `VDD_IN` is the usual one; carrier boards
/// vary and some report the module input under a different name.
const TOTAL_RAIL_NAMES: &[&str] = &["vdd_in", "vdd_sys_in", "sum of shunt voltages", "vdd_mux"];

/// Why a set of rails could not be turned into a power reading.
#[derive(Clone, Debug, PartialEq)]
pub enum RailError {
    /// No INA3221 present, or none readable by this user.
    NotFound { looked_in: Vec<String> },
    /// Rails were found, but none of them says it is the total.
    ///
    /// Deliberately fatal. The channels are nested, so there is no arithmetic — not a sum, not a
    /// maximum — that turns "some rails" into board power without knowing which is which.
    NoTotalRail { found: Vec<String> },
    /// A file existed and did not contain what its name promises.
    Unreadable { path: String, why: String },
}

impl core::fmt::Display for RailError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RailError::NotFound { looked_in } => write!(
                f,
                "no INA3221 rails found. Looked in: {}. On a Jetson these appear once the ina3221 \
                 driver is bound; on other hardware there may simply be no shunt monitor",
                looked_in.join(", ")
            ),
            RailError::NoTotalRail { found } => write!(
                f,
                "found rails {:?} but none is labelled as the board total, and these channels are \
                 NESTED rather than disjoint -- on a Jetson VDD_IN already contains VDD_CPU_GPU_CV \
                 and VDD_SOC, so summing them roughly doubles the answer. Name the total rail \
                 explicitly with Rails::with_total_label if you know which it is",
                found
            ),
            RailError::Unreadable { path, why } => write!(f, "{path}: {why}"),
        }
    }
}

/// One INA3221 channel.
#[derive(Clone, Debug, PartialEq)]
pub struct Rail {
    pub label: String,
    /// Where its reading comes from, already resolved to a layout.
    pub source: RailSource,
}

/// Which of the two driver layouts a rail's numbers come from.
#[derive(Clone, Debug, PartialEq)]
pub enum RailSource {
    /// Upstream hwmon: bus voltage in mV times current in mA.
    VoltsAmps { volt: PathBuf, curr: PathBuf },
    /// L4T downstream: milliwatts, directly.
    Milliwatts { power: PathBuf },
}

/// The rails on one INA3221, and which of them is the board total.
#[derive(Clone, Debug)]
pub struct Rails {
    rails: Vec<Rail>,
    total: usize,
}

/// Directories where an INA3221 turns up, most specific first.
fn search_roots() -> Vec<PathBuf> {
    vec![
        PathBuf::from("/sys/bus/i2c/drivers/ina3221"),
        PathBuf::from("/sys/bus/i2c/drivers/ina3221x"),
        PathBuf::from("/sys/class/hwmon"),
    ]
}

fn read_trim(p: &Path) -> Result<String, RailError> {
    std::fs::read_to_string(p)
        .map(|s| s.trim().to_string())
        .map_err(|e| RailError::Unreadable { path: p.display().to_string(), why: e.to_string() })
}

fn read_f64(p: &Path) -> Result<f64, RailError> {
    let s = read_trim(p)?;
    s.parse::<f64>()
        .map_err(|_| RailError::Unreadable { path: p.display().to_string(), why: format!("{s:?} is not a number") })
}

impl Rails {
    /// Find an INA3221 on this machine.
    pub fn detect() -> Result<Rails, RailError> {
        let roots = search_roots();
        for root in &roots {
            if let Ok(found) = Rails::at(root) {
                return Ok(found);
            }
        }
        Err(RailError::NotFound {
            looked_in: roots.iter().map(|p| p.display().to_string()).collect(),
        })
    }

    /// Find rails under an arbitrary directory.
    ///
    /// Public because it is the seam that makes every part of this testable without a Jetson: the
    /// tests build a directory of the right shape and point this at it. A backend whose only entry
    /// point hard-codes `/sys` is a backend that can only be tested by owning the hardware.
    pub fn at(root: &Path) -> Result<Rails, RailError> {
        let mut rails = Rails::scan(root)?;
        if rails.is_empty() {
            return Err(RailError::NotFound { looked_in: vec![root.display().to_string()] });
        }
        rails.sort_by(|a, b| a.label.cmp(&b.label));
        let total = rails
            .iter()
            .position(|r| TOTAL_RAIL_NAMES.contains(&r.label.to_ascii_lowercase().as_str()))
            .ok_or_else(|| RailError::NoTotalRail {
                found: rails.iter().map(|r| r.label.clone()).collect(),
            })?;
        Ok(Rails { rails, total })
    }

    /// Use the rail with this label as the board total.
    ///
    /// For boards whose total rail is named something this does not know. It is an override for a
    /// naming gap, not a way to opt out of the nesting problem: whatever you name here is treated
    /// as the whole board, so naming a subsidiary rail will under-report by however much the rest
    /// of the board draws.
    pub fn with_total_label(mut self, label: &str) -> Result<Rails, RailError> {
        let want = label.to_ascii_lowercase();
        self.total = self
            .rails
            .iter()
            .position(|r| r.label.to_ascii_lowercase() == want)
            .ok_or_else(|| RailError::NoTotalRail {
                found: self.rails.iter().map(|r| r.label.clone()).collect(),
            })?;
        Ok(self)
    }

    fn scan(root: &Path) -> Result<Vec<Rail>, RailError> {
        let mut out = Vec::new();
        // Depth is carried WITH each directory. Counting it as a single counter incremented per
        // directory popped -- which this first did -- is not depth at all: it stops descending
        // after six directories however shallow they are, so a device sitting behind a handful of
        // siblings becomes invisible. The bound exists because /sys is full of symlinks and an
        // unbounded walk there turns into a hang, so it has to be a real bound.
        const MAX_DEPTH: usize = 6;
        let mut stack = vec![(root.to_path_buf(), 0usize)];
        let mut seen = 0usize;
        while let Some((dir, depth)) = stack.pop() {
            // A second, absolute bound on total directories visited: depth alone does not stop a
            // symlink cycle that stays shallow while fanning out.
            seen += 1;
            if seen > 4096 {
                break;
            }
            let Ok(entries) = std::fs::read_dir(&dir) else { continue };
            let mut names: Vec<PathBuf> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
            names.sort();
            if depth < MAX_DEPTH {
                for p in &names {
                    // symlink_metadata, not is_dir(): is_dir() follows the link, and /sys is full
                    // of links pointing back up the tree.
                    if std::fs::symlink_metadata(p).map(|m| m.is_dir()).unwrap_or(false) {
                        stack.push((p.clone(), depth + 1));
                    }
                }
            }
            out.extend(Rails::rails_in(&dir)?);
        }
        Ok(out)
    }

    /// The rails declared directly in one directory, under either layout.
    fn rails_in(dir: &Path) -> Result<Vec<Rail>, RailError> {
        let mut out = Vec::new();

        // Upstream hwmon: in[123]_label / in[123]_input (mV) / curr[123]_input (mA).
        for ch in 1..=3 {
            let label_p = dir.join(format!("in{ch}_label"));
            let volt = dir.join(format!("in{ch}_input"));
            let curr = dir.join(format!("curr{ch}_input"));
            if label_p.exists() && volt.exists() && curr.exists() {
                out.push(Rail {
                    label: read_trim(&label_p)?,
                    source: RailSource::VoltsAmps { volt, curr },
                });
            }
        }

        // L4T downstream: rail_name_[012] / in_power[012]_input (mW).
        for ch in 0..=2 {
            let label_p = dir.join(format!("rail_name_{ch}"));
            let power = dir.join(format!("in_power{ch}_input"));
            if label_p.exists() && power.exists() {
                out.push(Rail { label: read_trim(&label_p)?, source: RailSource::Milliwatts { power } });
            }
        }
        Ok(out)
    }

    /// Every rail found, in label order. The total is [`Rails::total`].
    pub fn all(&self) -> &[Rail] {
        &self.rails
    }

    /// The rail treated as whole-board power.
    pub fn total(&self) -> &Rail {
        &self.rails[self.total]
    }

    /// Whole-board power, in **watts**.
    ///
    /// One rail, never a sum. See the module docs for why that is not a simplification.
    pub fn watts(&self) -> Result<f64, RailError> {
        match &self.total().source {
            // mV x mA = uW, so divide by 1e6 for watts. Getting this wrong by a factor of 1000 is
            // the easiest possible error here and would still look like a plausible wattage.
            RailSource::VoltsAmps { volt, curr } => Ok(read_f64(volt)? * read_f64(curr)? / 1.0e6),
            RailSource::Milliwatts { power } => Ok(read_f64(power)? / 1.0e3),
        }
    }
}

/// How often to poll. Matches the meter's own sampling interval.
pub const POLL: Duration = Duration::from_millis(200);

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a fixture directory shaped like one of the two drivers.
    fn write(dir: &Path, files: &[(&str, &str)]) {
        std::fs::create_dir_all(dir).unwrap();
        for (name, body) in files {
            std::fs::write(dir.join(name), body).unwrap();
        }
    }

    fn tmp(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("ft-ina3221-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    #[test]
    fn the_upstream_layout_reads_millivolts_times_milliamps_as_watts() {
        let d = tmp("upstream");
        write(
            &d,
            &[
                ("in1_label", "VDD_IN\n"),
                ("in1_input", "5000\n"),   // 5000 mV
                ("curr1_input", "2400\n"), // 2400 mA
                ("in2_label", "VDD_CPU_GPU_CV\n"),
                ("in2_input", "5000\n"),
                ("curr2_input", "1500\n"),
            ],
        );
        let r = Rails::at(&d).expect("a labelled total rail");
        assert_eq!(r.total().label, "VDD_IN");
        // 5000 mV x 2400 mA = 12_000_000 uW = 12 W. A factor-of-1000 slip still looks plausible,
        // which is why this asserts the value rather than a range.
        assert!((r.watts().unwrap() - 12.0).abs() < 1e-9, "{}", r.watts().unwrap());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn the_downstream_layout_reads_milliwatts() {
        let d = tmp("downstream");
        write(&d, &[("rail_name_0", "VDD_IN\n"), ("in_power0_input", "7350\n")]);
        let r = Rails::at(&d).expect("a labelled total rail");
        assert!((r.watts().unwrap() - 7.35).abs() < 1e-9, "{}", r.watts().unwrap());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn nested_rails_are_refused_rather_than_summed() {
        // The defect this module exists to avoid. Three rails, none labelled as the total: summing
        // them would report ~21 W for a board drawing 12, and look entirely reasonable doing it.
        let d = tmp("nototal");
        write(
            &d,
            &[
                ("in1_label", "VDD_CPU_GPU_CV\n"),
                ("in1_input", "5000\n"),
                ("curr1_input", "1500\n"),
                ("in2_label", "VDD_SOC\n"),
                ("in2_input", "5000\n"),
                ("curr2_input", "900\n"),
            ],
        );
        match Rails::at(&d) {
            Err(RailError::NoTotalRail { found }) => {
                assert_eq!(found.len(), 2, "it lists what it found: {found:?}");
                let msg = RailError::NoTotalRail { found }.to_string();
                assert!(msg.contains("NESTED"), "and says why summing is wrong: {msg}");
            }
            other => panic!("nested rails must be refused, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn an_unknown_total_rail_can_be_named_but_only_by_the_caller() {
        let d = tmp("override");
        write(&d, &[("rail_name_0", "BOARD_INPUT\n"), ("in_power0_input", "9000\n")]);
        // Not a name this knows, so it refuses...
        assert!(matches!(Rails::at(&d), Err(RailError::NoTotalRail { .. })));
        // ...until a caller who knows the board says which rail is the total. Reaching the override
        // needs a Rails, which `at` refused to build, so it is reached through the error path here
        // exactly as a caller would: scan, then name.
        let rails = Rails { rails: Rails::scan(&d).unwrap(), total: 0 };
        let named = rails.with_total_label("board_input").expect("case-insensitive");
        assert!((named.watts().unwrap() - 9.0).abs() < 1e-9);
        assert!(named.with_total_label("no_such_rail").is_err(), "and an unknown name is refused");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_file_that_is_not_a_number_says_which_file() {
        let d = tmp("garbage");
        write(&d, &[("rail_name_0", "VDD_IN\n"), ("in_power0_input", "not-a-number\n")]);
        let r = Rails::at(&d).expect("it is labelled");
        match r.watts() {
            Err(RailError::Unreadable { path, why }) => {
                assert!(path.contains("in_power0_input"), "names the file: {path}");
                assert!(why.contains("not a number"), "and what was wrong: {why}");
            }
            other => panic!("expected an Unreadable naming the file, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_device_nested_below_several_siblings_is_still_found() {
        // The walk used one counter incremented per directory popped and called it depth, so after
        // six directories it stopped descending -- and a device sitting behind a handful of
        // siblings, which is exactly how /sys/class/hwmon looks, became invisible. Shallow and wide,
        // not deep, is the shape that broke it.
        let root = tmp("wide");
        for i in 0..10 {
            std::fs::create_dir_all(root.join(format!("hwmon{i}"))).unwrap();
        }
        write(
            &root.join("hwmon9").join("device"),
            &[("rail_name_0", "VDD_IN\n"), ("in_power0_input", "4200\n")],
        );
        let r = Rails::at(&root).expect("found behind nine siblings");
        assert!((r.watts().unwrap() - 4.2).abs() < 1e-9);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn an_empty_tree_is_not_found_rather_than_zero_watts() {
        // Zero watts is a reading. "There is no sensor here" is not, and conflating them would put
        // a 0.0 into a ledger as though it had been measured.
        let d = tmp("empty");
        std::fs::create_dir_all(&d).unwrap();
        assert!(matches!(Rails::at(&d), Err(RailError::NotFound { .. })));
        let _ = std::fs::remove_dir_all(&d);
    }
}
