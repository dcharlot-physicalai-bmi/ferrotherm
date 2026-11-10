//! Head-to-head: ferrotherm and a pinned THRML runner, same workloads, one scorer.
//!
//! Accuracy is total variation and KL against an exact Boltzmann oracle, integrated
//! autocorrelation time and effective sample size on the energy trace, and the error
//! of the mean energy (and, for spins, the per-site magnetisation).
//!
//! Energy is a hardware counter or it is absent. Projected device prices are not read.

mod energy;
mod ft;
mod oracle;
mod parity;
mod score;
mod spec;

use energy::{
    EnergyTrial, GRADE_CONTENDED, GradeSummary, Joules, MIN_ENERGY_WINDOW_S, Probe, energy_rank,
    grade_for, poll_quiet, read_cpu_snap, snap_other_cpu, trial_joules, wait_for_quiet_host,
};
use ferrotherm_meter::Baseline;
use ft::Chain;
use score::{Score, mean_ci_95};
use spec::Workload;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

fn main() {
    let args = match Args::parse(std::env::args().skip(1)) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            eprintln!(
                "usage: ferrotherm-thrml-bench --out DIR [--trials N] [--warmup N] [--draws N] \
                 [--python PY] [--python-b PY] [--skip-thrml] [--contended] [--quiet-gate] [--scale] \
                 [--grid-scale] [--profile] [--profile-sweeps N] [--lever NAME] [--threads N] \
                 [--sides LIST] [--device auto|cpu|cuda] [--cpus LIST] [--lock PATH] \
                 [--lock-wait-s SECONDS] [--other-core-mean F] [--other-core-max F] \
                 [--foreign-cpu-mean F] [--foreign-cpu-max F]"
            );
            std::process::exit(2);
        }
    };
    if !args.cpus.is_empty() {
        energy::pin_current(&args.cpus).unwrap_or_else(|e| panic!("cpu pin: {e}"));
        eprintln!(
            "pinned cpus: {}",
            args.cpus
                .iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<_>>()
                .join(",")
        );
    }
    GRID_CHOICE.with(|c| c.set((args.levers, args.threads, args.gpu)));
    let work = if args.grid_scale {
        let mut grids = spec::grid_scaling(args.trials, args.warmup, args.draws);
        if !args.sides.is_empty() {
            grids.retain(|w| args.sides.contains(&w.open_grid().expect("grid id").0));
        }
        grids
    } else {
        let mut canonical = spec::canonical(args.trials, args.warmup, args.draws);
        if args.scale {
            canonical.extend(spec::scaling(args.trials, args.warmup, args.draws));
        }
        canonical
    };
    let spec_text = spec::write_spec(&work);
    fs::create_dir_all(&args.out).unwrap_or_else(|e| panic!("create {}: {e}", args.out.display()));
    let traces = args.out.join("traces");
    fs::create_dir_all(&traces).unwrap_or_else(|e| panic!("create {}: {e}", traces.display()));
    fs::write(args.out.join("spec.txt"), &spec_text).expect("write spec");

    let lock_wait = Instant::now();
    let Ok(host_lock) = energy::HostLock::acquire(&args.lock, args.lock_wait_s) else {
        write_lock_refusal(&args.out, &args.lock, lock_wait.elapsed().as_secs_f64());
        std::process::exit(1);
    };

    let mut probe = Probe::detect();
    eprintln!("wall: {}", probe.wall_note);
    eprintln!("rapl: {}", probe.rapl_note);
    eprintln!("rapl core: {}", probe.rapl_core_note);
    eprintln!("perf: {}", probe.perf_note);
    eprintln!("nvml: {}", probe.nvml_note);
    eprintln!("load1: {:?} quiet: {}", probe.load1, probe.quiet);

    let quiet_gate = if args.quiet_gate {
        eprintln!("quiet-gate preflight: {} s", energy::QUIET_GATE_SECONDS);
        let gate = energy::sample_quiet_gate(&args.cpus);
        let receipt = quiet_gate_json(&gate);
        fs::write(args.out.join("quiet-gate.json"), &receipt)
            .unwrap_or_else(|e| panic!("quiet-gate: {e}"));
        if !gate.held {
            let lock_json = release_lock_json(host_lock);
            let report = format!(
                "{{\n  \"schema\": \"ferrotherm-thrml-bench/1\",\n  \"evidence_grade\": {},\n  \
                 \"phases\": {phases},\n  \"host_lock\": {lock},\n  \
                 \"provenance\": {prov},\n  \"quiet_gate\": {gate},\n  \"energy\": null,\n  \
                 \"workloads\": []\n}}\n",
                json_str("load_gated"),
                phases = phases_json("load_gated", "not_run", "not_run"),
                lock = lock_json,
                prov = provenance_json(&probe, "load_gated"),
                gate = receipt,
            );
            fs::write(args.out.join("report.json"), &report).expect("write report");
            eprintln!("quiet gate refused the run; max load1 {:?}", gate.max_load1);
            std::process::exit(1);
        }
        Some(gate)
    } else {
        None
    };

    let held = quiet_gate.as_ref().is_some_and(|gate| gate.held);
    if args.profile {
        let spec_path = args.out.join("spec.txt");
        let accuracy = run_accuracy(
            &work,
            args.skip_thrml,
            &args.python,
            &args.python_b,
            &args.runner,
            &args.out,
            &spec_path,
            &traces,
            &args.device,
            args.grid_choice(),
            args.gpu,
        );
        let profile = profile_json(&work, args.grid_choice(), args.profile_sweeps, args.gpu);
        fs::write(args.out.join("profile.json"), &profile).expect("write profile");
        let parity_failed = accuracy.done.iter().any(|row| parity_blocks_exit(&row.4));
        let lock_json = release_lock_json(host_lock);
        let mut body = String::new();
        for (index, ft_trials, thrml_trials, thrml_b, parity) in &accuracy.done {
            body.push_str(&workload_json(
                &work[*index],
                ft_trials,
                thrml_trials.as_deref(),
                thrml_b.as_deref(),
                "measured",
                parity,
            ));
            body.push_str(",\n");
        }
        if body.ends_with(",\n") {
            body.truncate(body.len() - 2);
            body.push('\n');
        }
        let report = format!(
            "{{\n  \"schema\": \"ferrotherm-thrml-bench/1\",\n  \"evidence_grade\": \"not_run\",\n  \
             \"profile_note\": \"no energy was measured. wall_ns is VM or host elapsed time, not joules.\",\n  \
             \"host_lock\": {lock},\n  \"profile\": {profile},\n  \
             \"thrml_meta\": {meta},\n  \"thrml_b_meta\": {meta_b},\n  \
             \"thrml_error\": {err},\n  \"thrml_b_error\": {err_b},\n  \
             \"workloads\": [\n{body}  ]\n}}\n",
            lock = lock_json,
            meta = json_str(&accuracy.thrml_meta),
            meta_b = json_str(&accuracy.thrml_b_meta),
            err = if accuracy.thrml_note.is_empty() {
                "null".into()
            } else {
                json_str(&accuracy.thrml_note)
            },
            err_b = if accuracy.thrml_b_note.is_empty() {
                "null".into()
            } else {
                json_str(&accuracy.thrml_b_note)
            },
        );
        fs::write(args.out.join("report.json"), &report).expect("write report");
        if !accuracy.thrml_ok || !accuracy.thrml_b_ok || parity_failed {
            std::process::exit(1);
        }
        std::process::exit(0);
    }
    let order = energy::phase_order(held);
    let spec_path = args.out.join("spec.txt");
    let rule = args.window_rule();
    let (energy, grade, accuracy) = match order {
        ["energy", "accuracy"] => {
            eprintln!("phase order: energy, then accuracy");
            let (energy, grade) = energy_section(
                &mut probe,
                &work,
                args.contended,
                args.trials,
                &args.python,
                &args.python_b,
                &args.runner,
                &spec_path,
                !args.skip_thrml,
                &args.device,
                &args.cpus,
                held,
                &rule,
                args.grid_choice(),
                args.gpu,
            );
            let accuracy = run_accuracy(
                &work,
                args.skip_thrml,
                &args.python,
                &args.python_b,
                &args.runner,
                &args.out,
                &spec_path,
                &traces,
                &args.device,
                args.grid_choice(),
                args.gpu,
            );
            (energy, grade, accuracy)
        }
        ["accuracy", "energy"] => {
            eprintln!("phase order: accuracy, then energy");
            let accuracy = run_accuracy(
                &work,
                args.skip_thrml,
                &args.python,
                &args.python_b,
                &args.runner,
                &args.out,
                &spec_path,
                &traces,
                &args.device,
                args.grid_choice(),
                args.gpu,
            );
            let (energy, grade) = energy_section(
                &mut probe,
                &work,
                args.contended,
                args.trials,
                &args.python,
                &args.python_b,
                &args.runner,
                &spec_path,
                !args.skip_thrml && accuracy.thrml_ok,
                &args.device,
                &args.cpus,
                held,
                &rule,
                args.grid_choice(),
                args.gpu,
            );
            (energy, grade, accuracy)
        }
        other => panic!("phase_order returned {other:?}"),
    };
    let accuracy_grade = if accuracy.done.is_empty() {
        "not_run"
    } else {
        "measured"
    };
    let mut body = String::new();
    for (index, ft_trials, thrml_trials, thrml_b, parity) in &accuracy.done {
        body.push_str(&workload_json(
            &work[*index],
            ft_trials,
            thrml_trials.as_deref(),
            thrml_b.as_deref(),
            accuracy_grade,
            parity,
        ));
        body.push_str(",\n");
    }
    if body.ends_with(",\n") {
        body.truncate(body.len() - 2);
        body.push('\n');
    }
    let gate_json = match quiet_gate.as_ref() {
        Some(gate) => quiet_gate_json(gate),
        None => "null".into(),
    };
    let order_json = format!("[{}, {}]", json_str(order[0]), json_str(order[1]));
    let settle_note = "each idle bracket waits until load1 has stayed at or under 2 for 60 consecutive 1 Hz samples, bounded at 600 samples, then every 1 Hz sample through the bracket must stay at or under 2. Those samples are baseline_settle_load1 and each trial settle_load1. load1 is not the trial-window criterion.";
    let bundle = energy::bundle_grade(grade);
    let preflight_grade = if quiet_gate.is_none() {
        "not_run"
    } else {
        "quiet"
    };
    let lock_json = release_lock_json(host_lock);
    let report = format!(
        "{{\n  \"schema\": \"ferrotherm-thrml-bench/1\",\n  \"evidence_grade\": {grade},\n  \
         \"phases\": {phases},\n  \"host_lock\": {lock},\n  \
         \"phase_order\": {order},\n  \"settle_note\": {settle},\n  \
         \"provenance\": {prov},\n  \"quiet_gate\": {gate},\n  \
         \"energy\": {energy},\n  \"thrml_meta\": {meta},\n  \"thrml_b_meta\": {meta_b},\n  \
         \"thrml_error\": {err},\n  \"thrml_b_error\": {err_b},\n  \
         \"workloads\": [\n{body}  ]\n}}\n",
        grade = json_str(bundle),
        phases = phases_json(preflight_grade, grade, accuracy_grade),
        lock = lock_json,
        order = order_json,
        settle = json_str(settle_note),
        prov = provenance_json(&probe, bundle),
        gate = gate_json,
        meta = json_str(&accuracy.thrml_meta),
        meta_b = json_str(&accuracy.thrml_b_meta),
        err = if accuracy.thrml_note.is_empty() {
            "null".into()
        } else {
            json_str(&accuracy.thrml_note)
        },
        err_b = if accuracy.thrml_b_note.is_empty() {
            "null".into()
        } else {
            json_str(&accuracy.thrml_b_note)
        },
    );
    fs::write(args.out.join("report.json"), &report).expect("write report");
    eprintln!("wrote {}", args.out.join("report.json").display());
    let parity_failed = accuracy.done.iter().any(|row| parity_blocks_exit(&row.4));
    if !accuracy.thrml_ok || !accuracy.thrml_b_ok || parity_failed {
        std::process::exit(1);
    }
}

struct AccuracyRun {
    done: Vec<(
        usize,
        Vec<Chain>,
        Option<Vec<Chain>>,
        Option<Vec<Chain>>,
        Vec<(String, bool, String)>,
    )>,
    thrml_ok: bool,
    thrml_note: String,
    thrml_meta: String,
    thrml_b_ok: bool,
    thrml_b_note: String,
    thrml_b_meta: String,
}

/// A `THRML_LIMIT` row is the baseline's own ceiling. It refuses the comparison and it does
/// not fail the sizes that ran.
fn parity_blocks_exit(rows: &[(String, bool, String)]) -> bool {
    rows.iter()
        .any(|(_, ok, reason)| !*ok && !reason.starts_with("THRML_LIMIT"))
}

fn thrml_limit_reason(kind: &str) -> String {
    format!(
        "THRML_LIMIT {kind}: the THRML process was SIGKILLed. This is THRML's memory limit, not a ferrotherm result."
    )
}

fn run_accuracy(
    work: &[Workload],
    skip_thrml: bool,
    python: &Path,
    python_b: &Path,
    runner: &Path,
    out: &Path,
    spec: &Path,
    traces: &Path,
    device: &str,
    grid: Option<ft::GridChoice>,
    gpu: bool,
) -> AccuracyRun {
    let mut done = Vec::new();
    let (thrml_ok, thrml_note, thrml_meta, thrml_by_id, limits_a) = if skip_thrml {
        (
            true,
            "THRML runner skipped on request".to_string(),
            String::new(),
            None,
            std::collections::BTreeMap::new(),
        )
    } else {
        load_baseline(
            python,
            runner,
            spec,
            &out.join("thrml_raw.txt"),
            device,
            "A",
            "baseline A",
            work,
        )
    };
    let (thrml_b_ok, thrml_b_note, thrml_b_meta, thrml_b_by_id, limits_b) = if skip_thrml {
        (
            true,
            "THRML runner skipped on request".to_string(),
            String::new(),
            None,
            std::collections::BTreeMap::new(),
        )
    } else if python_b.as_os_str().is_empty() {
        (
            false,
            "baseline B interpreter was not set. Pass --python-b or THRML_PYTHON_B. A is not used in its place.".into(),
            String::new(),
            None,
            std::collections::BTreeMap::new(),
        )
    } else {
        load_baseline(
            python_b,
            runner,
            spec,
            &out.join("thrml_b_raw.txt"),
            device,
            "B",
            "baseline B",
            work,
        )
    };
    let mut thrml_ok = thrml_ok;
    let mut thrml_note = thrml_note;
    let mut thrml_b_ok = thrml_b_ok;
    let mut thrml_b_note = thrml_b_note;
    for w in work {
        eprintln!("ferrotherm {}", w.id);
        let mut ft_trials = Vec::new();
        for t in 0..w.trials {
            let seed = w.seed0 + t as u64;
            let chain = if gpu {
                run_gpu_chain(w, seed).unwrap_or_else(|e| panic!("{} seed {seed}: {e}", w.id))
            } else {
                ft::run_with(w, seed, grid).unwrap_or_else(|e| panic!("{} seed {seed}: {e}", w.id))
            };
            if t == 0 {
                write_trace(traces, w.id, "ferrotherm", &chain);
            }
            eprintln!(
                "  seed {seed} tv {:.4} tau {:.3} ess/s {}",
                chain.score.tv,
                chain.score.tau_int_energy,
                fmt_opt(chain.score.ess_per_second)
            );
            ft_trials.push(chain);
        }
        let thrml_trials = if limits_a.contains_key(w.id) {
            None
        } else {
            take_side(w, &thrml_by_id, &mut thrml_note, &mut thrml_ok)
        };
        let thrml_b_trials = if limits_b.contains_key(w.id) {
            None
        } else {
            take_side(w, &thrml_b_by_id, &mut thrml_b_note, &mut thrml_b_ok)
        };
        if let Some(trials) = &thrml_trials
            && let Some(first) = trials.first()
        {
            write_trace(traces, w.id, "thrml", first);
        }
        if let Some(trials) = &thrml_b_trials
            && let Some(first) = trials.first()
        {
            write_trace(traces, w.id, spec::THRML_B_SIDE, first);
        }
        let mut parity = parity_for(
            w,
            &ft_trials,
            thrml_trials.as_deref(),
            thrml_b_trials.as_deref(),
        );
        if let Some(kind) = limits_a.get(w.id) {
            parity.push(("thrml".into(), false, thrml_limit_reason(kind)));
        }
        if let Some(kind) = limits_b.get(w.id) {
            parity.push((spec::THRML_B_SIDE.into(), false, thrml_limit_reason(kind)));
        }
        for (side, ok, reason) in &parity {
            if !ok {
                eprintln!("parity refused {side} {}: {reason}", w.id);
            }
        }
        done.push((
            work.iter()
                .position(|item| item.id == w.id)
                .expect("workload id"),
            ft_trials,
            thrml_trials,
            thrml_b_trials,
            parity,
        ));
    }
    AccuracyRun {
        done,
        thrml_ok,
        thrml_note,
        thrml_meta,
        thrml_b_ok,
        thrml_b_note,
        thrml_b_meta,
    }
}

fn load_baseline(
    python: &Path,
    runner: &Path,
    spec: &Path,
    out: &Path,
    device: &str,
    baseline: &str,
    label: &str,
    work: &[Workload],
) -> (
    bool,
    String,
    String,
    Option<std::collections::BTreeMap<String, Vec<RawChain>>>,
    std::collections::BTreeMap<String, String>,
) {
    let split = !work.is_empty() && work.iter().all(|w| w.open_grid().is_some());
    if !split {
        return match run_thrml(python, runner, spec, out, device, baseline) {
            Ok(()) => match fs::read_to_string(out)
                .ok()
                .as_deref()
                .map(parse_thrml)
                .transpose()
            {
                Ok(Some((meta, map, limits))) => (true, String::new(), meta, Some(map), limits),
                Ok(None) => (
                    false,
                    format!("{label} produced no output"),
                    String::new(),
                    None,
                    std::collections::BTreeMap::new(),
                ),
                Err(e) => (
                    false,
                    format!("{label}: {e}"),
                    String::new(),
                    None,
                    std::collections::BTreeMap::new(),
                ),
            },
            Err(e) => (
                false,
                format!("{label}: {e}"),
                String::new(),
                None,
                std::collections::BTreeMap::new(),
            ),
        };
    }
    // One process per side. A SIGKILL on side 4096 must not throw away sides that already
    // finished, and the kill is THRML's memory limit, not a ferrotherm result.
    let mut merged: std::collections::BTreeMap<String, Vec<RawChain>> =
        std::collections::BTreeMap::new();
    let mut limits = std::collections::BTreeMap::new();
    let mut meta = String::new();
    let mut notes = Vec::new();
    let mut ok = true;
    let mut raw_all = String::new();
    let stem = out.file_stem().and_then(|s| s.to_str()).unwrap_or("thrml");
    let dir = out.parent().unwrap_or_else(|| Path::new("."));
    for w in work {
        let spec_path = dir.join(format!("{stem}.{}.spec", w.id));
        let one_out = dir.join(format!("{stem}.{}.txt", w.id));
        if let Err(e) = fs::write(&spec_path, spec::write_spec(std::slice::from_ref(w))) {
            ok = false;
            notes.push(format!("{label} {}: could not write spec: {e}", w.id));
            continue;
        }
        match run_thrml_status(python, runner, &spec_path, &one_out, device, baseline) {
            Err(e) => {
                ok = false;
                notes.push(format!("{label} {}: {e}", w.id));
            }
            Ok((status, _err_text)) if status.success() => match fs::read_to_string(&one_out) {
                Ok(text) => {
                    raw_all.push_str(&text);
                    if !raw_all.ends_with('\n') {
                        raw_all.push('\n');
                    }
                    match parse_thrml(&text) {
                        Ok((m, map, lim)) => {
                            if meta.is_empty() {
                                meta = m;
                            }
                            for (k, v) in map {
                                merged.entry(k).or_default().extend(v);
                            }
                            limits.extend(lim);
                        }
                        Err(e) => {
                            ok = false;
                            notes.push(format!("{label} {}: {e}", w.id));
                        }
                    }
                }
                Err(e) => {
                    ok = false;
                    notes.push(format!("{label} {}: {e}", w.id));
                }
            },
            Ok((status, err_text)) => {
                if limit_from_status(&status).is_some() || thrml_oom_text(&err_text) {
                    limits.insert(w.id.to_string(), "oom".to_string());
                    raw_all.push_str(&format!("LIMIT {} oom\n", w.id));
                    eprintln!("{label} {} THRML_LIMIT oom: {status}", w.id);
                } else {
                    ok = false;
                    notes.push(format!("{label} {} exited {status}", w.id));
                }
            }
        }
    }
    let _ = fs::write(out, raw_all);
    (ok, notes.join("; "), meta, Some(merged), limits)
}

/// `137` is `128 + SIGKILL`. Either the signal or that code is the kernel's OOM kill.
fn limit_from_exit(code: Option<i32>, signal: Option<i32>) -> Option<&'static str> {
    if signal == Some(9) || code == Some(137) {
        Some("oom")
    } else {
        None
    }
}

fn limit_from_status(status: &std::process::ExitStatus) -> Option<&'static str> {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        limit_from_exit(status.code(), status.signal())
    }
    #[cfg(not(unix))]
    {
        limit_from_exit(status.code(), None)
    }
}

fn take_side(
    w: &Workload,
    by_id: &Option<std::collections::BTreeMap<String, Vec<RawChain>>>,
    note: &mut String,
    ok: &mut bool,
) -> Option<Vec<Chain>> {
    match by_id {
        Some(map) => match map.get(w.id) {
            Some(raws) => Some(score_thrml(w, raws)),
            None => {
                *note = format!("THRML raw output has no workload {}", w.id);
                *ok = false;
                None
            }
        },
        None => None,
    }
}

fn parity_for(
    w: &Workload,
    ft: &[Chain],
    a: Option<&[Chain]>,
    b: Option<&[Chain]>,
) -> Vec<(String, bool, String)> {
    let Some((side, _)) = w.open_grid() else {
        return Vec::new();
    };
    let gap = grid_gap16();
    let onsager = ferrotherm::free_energy::onsager_energy_density(w.beta, 1.0);
    let mut out = Vec::new();
    out.push(judge_side(
        "ferrotherm",
        side,
        ft,
        w.exact_energy,
        onsager,
        gap,
    ));
    if let Some(chains) = a {
        out.push(judge_side(
            "thrml",
            side,
            chains,
            w.exact_energy,
            onsager,
            gap,
        ));
    }
    if let Some(chains) = b {
        out.push(judge_side(
            spec::THRML_B_SIDE,
            side,
            chains,
            w.exact_energy,
            onsager,
            gap,
        ));
    }
    out
}

fn judge_side(
    name: &str,
    side: usize,
    chains: &[Chain],
    exact: Option<f64>,
    onsager: f64,
    gap16: f64,
) -> (String, bool, String) {
    let Some(first) = chains.first() else {
        return (name.into(), false, format!("{name} produced no chain"));
    };
    let other = chains.get(1).map(|c| c.score.mean_energy);
    let verdict =
        parity::judge_ising_open(side, first.score.mean_energy, other, exact, onsager, gap16);
    (name.into(), verdict.ok, verdict.reason)
}

fn grid_gap16() -> f64 {
    let tm = oracle::ising_open_transfer(16, 1.0, 0.3)
        .expect("16x16 transfer matrix")
        .mean_energy;
    let onsager = ferrotherm::free_energy::onsager_energy_density(0.3, 1.0);
    (tm / 256.0 - onsager).abs()
}

#[cfg(feature = "gpu")]
fn with_gpu<T>(f: impl FnOnce(&ferrotherm_gpu::Gpu) -> Result<T, String>) -> Result<T, String> {
    thread_local! {
        static GPU: std::cell::RefCell<Option<ferrotherm_gpu::Gpu>> =
            const { std::cell::RefCell::new(None) };
    }
    GPU.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            let gpu = ferrotherm_gpu::Gpu::new().ok_or_else(|| {
                "no wgpu adapter. The GPU lever is not run on the CPU.".to_string()
            })?;
            if !gpu.is_hardware() {
                eprintln!(
                    "wgpu adapter {} is not hardware. Timings from it are not a GPU result.",
                    gpu.adapter().name
                );
            }
            *slot = Some(gpu);
        }
        f(slot.as_ref().expect("gpu cached"))
    })
}

fn run_gpu_chain(w: &Workload, seed: u64) -> Result<Chain, String> {
    #[cfg(feature = "gpu")]
    {
        let (side, j) = w
            .open_grid()
            .ok_or_else(|| format!("{} is not an open grid", w.id))?;
        let thin = u32::try_from(w.thin.max(1)).map_err(|_| "thin does not fit u32")?;
        let warmup = u32::try_from(w.warmup).map_err(|_| "warmup does not fit u32")?;
        let draws = u32::try_from(w.draws).map_err(|_| "draws does not fit u32")?;
        with_gpu(|gpu| {
            let mut state = vec![-1i8; side * side];
            let t0 = std::time::Instant::now();
            // One submission for warmup and every kept sample. The bond sum is read once.
            let (energies, timing) = gpu.sweep_packed_energies(
                side, j, w.beta, &mut state, seed, 0, warmup, draws, thin,
            )?;
            let seconds = t0.elapsed().as_secs_f64();
            eprintln!(
                "gpu side {side} submits {} dispatches {} sweeps/submit {:.1} setup_ns {} submit_ns {}",
                timing.submits,
                timing.dispatches,
                timing.sweeps_per_submit(),
                timing.setup_ns,
                timing.submit_ns
            );
            let mut chain = ft::score_energies(w, seconds, None, &energies)?;
            chain.submits = timing.submits;
            chain.sweeps_per_submit = timing.sweeps_per_submit();
            Ok(chain)
        })
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = (w, seed);
        Err(
            "the bench was built without the gpu feature. Rebuild with --features gpu. \
             The GPU lever is not run on the CPU."
                .into(),
        )
    }
}

struct Args {
    out: PathBuf,
    trials: usize,
    warmup: usize,
    draws: usize,
    python: PathBuf,
    python_b: PathBuf,
    runner: PathBuf,
    skip_thrml: bool,
    contended: bool,
    quiet_gate: bool,
    scale: bool,
    grid_scale: bool,
    profile: bool,
    profile_sweeps: usize,
    levers: ferrotherm::grid::Levers,
    gpu: bool,
    threads: usize,
    sides: Vec<usize>,
    device: String,
    cpus: Vec<usize>,
    lock: PathBuf,
    lock_wait_s: f64,
    other_core_mean: f64,
    other_core_max: f64,
    foreign_cpu_mean: f64,
    foreign_cpu_max: f64,
}

impl Args {
    fn parse(args: impl Iterator<Item = String>) -> Result<Args, String> {
        let mut out = None;
        let mut trials = spec::TRIALS;
        let mut warmup = spec::WARMUP;
        let mut draws = spec::DRAWS;
        let mut python = std::env::var("THRML_PYTHON").unwrap_or_else(|_| "python3".into());
        let mut python_b = std::env::var("THRML_PYTHON_B").unwrap_or_default();
        let mut skip = false;
        let mut contended = false;
        let mut quiet_gate = false;
        let mut scale = false;
        let mut grid_scale = false;
        let mut profile = false;
        let mut profile_sweeps = 4usize;
        let mut levers = ferrotherm::grid::Levers::SCALAR;
        let mut gpu = false;
        let mut threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let mut sides = Vec::new();
        let mut device = "auto".to_string();
        let mut cpus = String::new();
        let rule = energy::TrialWindowRule::defaults();
        let mut lock = energy::HOST_LOCK_DEFAULT.to_string();
        let mut lock_wait_s = energy::HOST_LOCK_WAIT_S;
        let mut other_core_mean = rule.other_core_mean_max;
        let mut other_core_max = rule.other_core_sample_max;
        let mut foreign_cpu_mean = rule.foreign_cpu_mean_max;
        let mut foreign_cpu_max = rule.foreign_cpu_sample_max;
        let mut it = args;
        while let Some(a) = it.next() {
            match a.as_str() {
                "--out" => out = Some(need(&mut it, "--out")?),
                "--trials" => trials = parse_usize(&need(&mut it, "--trials")?, "--trials")?,
                "--warmup" => warmup = parse_usize(&need(&mut it, "--warmup")?, "--warmup")?,
                "--draws" => draws = parse_usize(&need(&mut it, "--draws")?, "--draws")?,
                "--python" => python = need(&mut it, "--python")?,
                "--python-b" => python_b = need(&mut it, "--python-b")?,
                "--skip-thrml" => skip = true,
                "--contended" => contended = true,
                "--quiet-gate" => quiet_gate = true,
                "--scale" => scale = true,
                "--grid-scale" => grid_scale = true,
                "--profile" => profile = true,
                "--profile-sweeps" => {
                    profile_sweeps =
                        parse_usize(&need(&mut it, "--profile-sweeps")?, "--profile-sweeps")?;
                }
                "--lever" => {
                    let name = need(&mut it, "--lever")?;
                    if name == "gpu" {
                        if levers != ferrotherm::grid::Levers::SCALAR {
                            return Err(
                                "gpu does not combine with a CPU lever. Run it on its own.".into(),
                            );
                        }
                        gpu = true;
                    } else if gpu {
                        return Err(format!(
                            "--lever {name} does not combine with gpu. Run that lever on its own."
                        ));
                    } else {
                        let one = ferrotherm::grid::Levers::one(&name)?;
                        levers.parallel |= one.parallel;
                        levers.simd |= one.simd;
                        levers.tiled |= one.tiled;
                        levers.sparse_rng |= one.sparse_rng;
                        levers.incremental |= one.incremental;
                    }
                }
                "--threads" => threads = parse_usize(&need(&mut it, "--threads")?, "--threads")?,
                "--sides" => {
                    let text = need(&mut it, "--sides")?;
                    for part in text.split(',') {
                        let side = parse_usize(part.trim(), "--sides")?;
                        if !spec::GRID_SIDES.contains(&side) {
                            return Err(format!(
                                "--sides got {side}, which is not one of {}",
                                spec::GRID_SIDES.map(|s| s.to_string()).join(",")
                            ));
                        }
                        sides.push(side);
                    }
                }
                "--device" => device = need(&mut it, "--device")?,
                "--cpus" => cpus = need(&mut it, "--cpus")?,
                "--lock" => lock = need(&mut it, "--lock")?,
                "--lock-wait-s" => {
                    lock_wait_s = parse_f64(&need(&mut it, "--lock-wait-s")?, "--lock-wait-s")?;
                }
                "--other-core-mean" => {
                    other_core_mean =
                        parse_f64(&need(&mut it, "--other-core-mean")?, "--other-core-mean")?;
                }
                "--other-core-max" => {
                    other_core_max =
                        parse_f64(&need(&mut it, "--other-core-max")?, "--other-core-max")?;
                }
                "--foreign-cpu-mean" => {
                    foreign_cpu_mean =
                        parse_f64(&need(&mut it, "--foreign-cpu-mean")?, "--foreign-cpu-mean")?;
                }
                "--foreign-cpu-max" => {
                    foreign_cpu_max =
                        parse_f64(&need(&mut it, "--foreign-cpu-max")?, "--foreign-cpu-max")?;
                }
                "--help" | "-h" => return Err("help".into()),
                other => return Err(format!("unknown argument {other}")),
            }
        }
        let out = out.ok_or("missing --out")?;
        if trials < 2 {
            return Err("--trials must be at least 2 so a confidence interval exists".into());
        }
        if draws < 64 {
            return Err(
                "--draws must be at least 64; tau_int is NaN under 16 and noisy near it".into(),
            );
        }
        if !matches!(device.as_str(), "auto" | "cpu" | "cuda") {
            return Err(format!("--device wants auto, cpu, or cuda, got {device}"));
        }
        if quiet_gate && contended {
            return Err("--quiet-gate and --contended name different evidence grades".into());
        }
        if scale && grid_scale {
            return Err("--scale and --grid-scale are different workload sets".into());
        }
        if profile && !grid_scale {
            return Err("--profile is the no-energy sub-mode of --grid-scale".into());
        }
        if grid_scale && !profile && cpus.is_empty() {
            return Err(
                "--cpus is required for a grid-scale energy run. There is no default list. \
                 The host driver passes the cores it is measuring."
                    .into(),
            );
        }
        if !sides.is_empty() && !grid_scale {
            return Err("--sides filters --grid-scale and was given without it".into());
        }
        if threads == 0 {
            return Err("--threads must be at least 1".into());
        }
        if profile_sweeps == 0 {
            return Err("--profile-sweeps must be at least 1".into());
        }
        if gpu && !cfg!(feature = "gpu") {
            return Err(
                "--lever gpu needs this bench rebuilt with --features gpu. It is not run on the CPU."
                    .into(),
            );
        }
        let cpus = if cpus.is_empty() {
            Vec::new()
        } else {
            energy::parse_cpu_list(&cpus)?
        };
        Ok(Args {
            out: PathBuf::from(out),
            trials,
            warmup,
            draws,
            python: PathBuf::from(python),
            python_b: PathBuf::from(python_b),
            runner: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python/runner.py"),
            skip_thrml: skip,
            contended,
            quiet_gate,
            scale,
            grid_scale,
            profile,
            profile_sweeps,
            levers,
            gpu,
            threads,
            sides,
            device,
            cpus,
            lock: PathBuf::from(lock),
            lock_wait_s,
            other_core_mean,
            other_core_max,
            foreign_cpu_mean,
            foreign_cpu_max,
        })
    }
}

fn parse_f64(text: &str, flag: &str) -> Result<f64, String> {
    let value: f64 = text
        .parse()
        .map_err(|_| format!("{flag} wants a number, got {text}"))?;
    if !value.is_finite() || value < 0.0 {
        return Err(format!(
            "{flag} wants a finite number at or above 0, got {text}"
        ));
    }
    Ok(value)
}

impl Args {
    fn grid_choice(&self) -> Option<ft::GridChoice> {
        if self.gpu {
            None
        } else {
            Some(ft::GridChoice {
                levers: self.levers,
                threads: self.threads,
            })
        }
    }

    fn window_rule(&self) -> energy::TrialWindowRule {
        energy::TrialWindowRule {
            enforce: true,
            other_core_mean_max: self.other_core_mean,
            other_core_sample_max: self.other_core_max,
            foreign_cpu_mean_max: self.foreign_cpu_mean,
            foreign_cpu_sample_max: self.foreign_cpu_max,
        }
    }
}

fn need(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    it.next().ok_or_else(|| format!("{flag} needs a value"))
}

fn parse_usize(s: &str, flag: &str) -> Result<usize, String> {
    s.parse()
        .map_err(|_| format!("{flag} wants an integer, got {s}"))
}

fn run_thrml(
    python: &Path,
    runner: &Path,
    spec: &Path,
    out: &Path,
    device: &str,
    baseline: &str,
) -> Result<(), String> {
    let (status, err_text) = run_thrml_status(python, runner, spec, out, device, baseline)?;
    if status.success() {
        Ok(())
    } else if limit_from_status(&status).is_some() || thrml_oom_text(&err_text) {
        Err(thrml_limit_reason("oom"))
    } else {
        Err(format!("THRML runner exited {status}"))
    }
}

fn run_thrml_status(
    python: &Path,
    runner: &Path,
    spec: &Path,
    out: &Path,
    device: &str,
    baseline: &str,
) -> Result<(std::process::ExitStatus, String), String> {
    eprintln!(
        "THRML via {} {} --device {device}",
        python.display(),
        runner.display()
    );
    let mut child = Command::new(python)
        .arg(runner)
        .arg("--spec")
        .arg(spec)
        .arg("--out")
        .arg(out)
        .arg("--device")
        .arg(device)
        .arg("--baseline")
        .arg(baseline)
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not start {}: {e}", python.display()))?;
    let mut err_pipe = child.stderr.take().ok_or("THRML runner has no stderr")?;
    let drained = std::thread::spawn(move || {
        let mut buf = String::new();
        use std::io::Read;
        let _ = err_pipe.read_to_string(&mut buf);
        buf
    });
    let status = child
        .wait()
        .map_err(|e| format!("THRML runner wait: {e}"))?;
    let err_text = drained.join().unwrap_or_default();
    if !err_text.is_empty() {
        eprint!("{err_text}");
    }
    Ok((status, err_text))
}

/// JAX on CUDA reports a failed allocation as text and exit code 1. The kernel OOM killer
/// is signal 9 or code 137. Either one is THRML's memory limit, on CPU and on CUDA.
fn thrml_oom_text(text: &str) -> bool {
    let t = text.to_ascii_lowercase();
    t.contains("resource_exhausted")
        || t.contains("out of memory")
        || t.contains("cuda_error_out_of_memory")
        || t.contains("cublas_status_alloc_failed")
        || t.contains("failed to allocate")
}

struct RawChain {
    seed: u64,
    seconds: f64,
    jit_seconds: f64,
    states_spin: Vec<Vec<i8>>,
    states_potts: Vec<Vec<u8>>,
    energies: Vec<f64>,
}

fn parse_thrml(
    text: &str,
) -> Result<
    (
        String,
        std::collections::BTreeMap<String, Vec<RawChain>>,
        std::collections::BTreeMap<String, String>,
    ),
    String,
> {
    let mut meta = String::new();
    let mut map: std::collections::BTreeMap<String, Vec<RawChain>> =
        std::collections::BTreeMap::new();
    let mut limits: std::collections::BTreeMap<String, String> = std::collections::BTreeMap::new();
    let mut cur: Option<RawChainBuild> = None;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("META ") {
            meta = rest.to_string();
        } else if let Some(rest) = line.strip_prefix("WORK ") {
            if cur.is_some() {
                return Err("WORK inside an open WORK".into());
            }
            cur = Some(RawChainBuild::parse(rest)?);
        } else if let Some(rest) = line.strip_prefix("ENERGY ") {
            let c = cur.as_mut().ok_or("ENERGY outside WORK")?;
            let v: f64 = rest
                .trim()
                .parse()
                .map_err(|_| format!("bad ENERGY {rest}"))?;
            c.energies.push(v);
        } else if let Some(rest) = line.strip_prefix("STATE ") {
            let c = cur.as_mut().ok_or("STATE outside WORK")?;
            c.push_state(rest)?;
        } else if let Some(rest) = line.strip_prefix("LIMIT ") {
            if cur.is_some() {
                return Err("LIMIT inside an open WORK".into());
            }
            let mut parts = rest.split_whitespace();
            let id = parts.next().ok_or("LIMIT without a workload id")?;
            let kind = parts.next().unwrap_or("oom");
            if kind != "oom" {
                return Err(format!(
                    "LIMIT {id} has kind {kind}; the recorded kind is oom"
                ));
            }
            limits.insert(id.to_string(), kind.to_string());
        } else if line == "END" {
            let c = cur.take().ok_or("END outside WORK")?;
            map.entry(c.id.clone()).or_default().push(c.finish()?);
        } else if !line.is_empty() {
            return Err(format!("bad THRML line: {line}"));
        }
    }
    if cur.is_some() {
        return Err("THRML output ended inside a workload".into());
    }
    Ok((meta, map, limits))
}

struct RawChainBuild {
    id: String,
    seed: u64,
    kind: String,
    seconds: f64,
    jit_seconds: f64,
    rows: Vec<Vec<i64>>,
    energies: Vec<f64>,
}

impl RawChainBuild {
    fn parse(rest: &str) -> Result<Self, String> {
        let mut id = None;
        let mut seed = None;
        let mut kind = None;
        let mut seconds = None;
        let mut jit = None;
        for part in rest.split_whitespace() {
            let (k, v) = part
                .split_once('=')
                .ok_or_else(|| format!("bad field {part}"))?;
            match k {
                "id" => id = Some(v.to_string()),
                "seed" => seed = Some(v.parse().map_err(|_| format!("bad seed {v}"))?),
                "kind" => kind = Some(v.to_string()),
                "seconds" => seconds = Some(v.parse().map_err(|_| format!("bad seconds {v}"))?),
                "jit_seconds" => jit = Some(v.parse().map_err(|_| format!("bad jit {v}"))?),
                _ => {}
            }
        }
        Ok(RawChainBuild {
            id: id.ok_or("WORK missing id")?,
            seed: seed.ok_or("WORK missing seed")?,
            kind: kind.ok_or("WORK missing kind")?,
            seconds: seconds.ok_or("WORK missing seconds")?,
            jit_seconds: jit.ok_or("WORK missing jit_seconds")?,
            rows: Vec::new(),
            energies: Vec::new(),
        })
    }

    fn push_state(&mut self, rest: &str) -> Result<(), String> {
        let mut row = Vec::new();
        for tok in rest.split_whitespace() {
            row.push(
                tok.parse::<i64>()
                    .map_err(|_| format!("bad state token {tok}"))?,
            );
        }
        self.rows.push(row);
        Ok(())
    }

    fn finish(self) -> Result<RawChain, String> {
        if self.rows.is_empty() && self.energies.is_empty() {
            return Err(format!("{} produced no states and no energies", self.id));
        }
        if self.rows.is_empty() {
            return Ok(RawChain {
                seed: self.seed,
                seconds: self.seconds,
                jit_seconds: self.jit_seconds,
                states_spin: Vec::new(),
                states_potts: Vec::new(),
                energies: self.energies,
            });
        }
        let width = self.rows[0].len();
        let mut spins = Vec::new();
        let mut potts = Vec::new();
        for row in &self.rows {
            if row.len() != width {
                return Err(format!("{} changed width mid-chain", self.id));
            }
            if self.kind == "potts" {
                let mut s = Vec::with_capacity(width);
                for &v in row {
                    if !(0..=255).contains(&v) {
                        return Err(format!("{} potts state {v} is not a byte", self.id));
                    }
                    s.push(v as u8);
                }
                potts.push(s);
            } else {
                let mut s = Vec::with_capacity(width);
                for &v in row {
                    if v != 1 && v != -1 {
                        return Err(format!("{} spin {v} is not ±1", self.id));
                    }
                    s.push(v as i8);
                }
                spins.push(s);
            }
        }
        Ok(RawChain {
            seed: self.seed,
            seconds: self.seconds,
            jit_seconds: self.jit_seconds,
            states_spin: spins,
            states_potts: potts,
            energies: Vec::new(),
        })
    }
}

fn score_thrml(w: &Workload, raws: &[RawChain]) -> Vec<Chain> {
    raws.iter()
        .map(|raw| {
            let chain = score_external(w, raw).unwrap_or_else(|e| panic!("{}: {e}", w.id));
            eprintln!(
                "  thrml seed {} tv {:.4} tau {:.3} ess/s {} (jit {:.2}s, not in ess/s)",
                raw.seed,
                chain.score.tv,
                chain.score.tau_int_energy,
                fmt_opt(chain.score.ess_per_second),
                raw.jit_seconds
            );
            chain
        })
        .collect()
}

fn score_external(w: &Workload, raw: &RawChain) -> Result<Chain, String> {
    if raw.states_spin.is_empty() && raw.states_potts.is_empty() && !raw.energies.is_empty() {
        return ft::score_energies(w, raw.seconds, Some(raw.jit_seconds), &raw.energies);
    }
    // Re-score with the ferrotherm energy and the ferrotherm oracle. The external
    // runner contributed states and a wall-clock; it did not contribute the metric.
    ft::score_observed(
        w,
        raw.seconds,
        Some(raw.jit_seconds),
        &raw.states_spin,
        &raw.states_potts,
    )
}

fn write_trace(dir: &Path, id: &str, side: &str, chain: &Chain) {
    let energy_path = dir.join(format!("{id}_{side}_trial0.energy.txt"));
    let mut e = String::new();
    for v in &chain.energies {
        e.push_str(&format!("{v:.16}\n"));
    }
    fs::write(&energy_path, e).expect("energy trace");
    let state_path = dir.join(format!("{id}_{side}_trial0.states.txt"));
    let mut s = String::new();
    if !chain.states_spin.is_empty() {
        for row in &chain.states_spin {
            for (i, v) in row.iter().enumerate() {
                if i > 0 {
                    s.push(' ');
                }
                s.push_str(&format!("{v}"));
            }
            s.push('\n');
        }
    } else {
        for row in &chain.states_potts {
            for (i, v) in row.iter().enumerate() {
                if i > 0 {
                    s.push(' ');
                }
                s.push_str(&format!("{v}"));
            }
            s.push('\n');
        }
    }
    fs::write(&state_path, s).expect("state trace");
}

fn workload_json(
    w: &Workload,
    ft_trials: &[Chain],
    thrml: Option<&[Chain]>,
    thrml_b: Option<&[Chain]>,
    grade: &str,
    parity: &[(String, bool, String)],
) -> String {
    let mut s = String::new();
    s.push_str("    {\n");
    s.push_str(&format!("      \"id\": {},\n", json_str(w.id)));
    s.push_str(&format!("      \"evidence_grade\": {},\n", json_str(grade)));
    s.push_str(&format!("      \"kind\": {},\n", json_str(w.kind.as_str())));
    s.push_str(&format!("      \"source\": {},\n", json_str(w.source)));
    s.push_str(&format!("      \"beta\": {:.16},\n", w.beta));
    s.push_str(&format!("      \"n\": {},\n", w.n));
    s.push_str(&format!("      \"q\": {},\n", w.q));
    s.push_str(&format!("      \"warmup\": {},\n", w.warmup));
    s.push_str(&format!("      \"draws\": {},\n", w.draws));
    s.push_str(&format!("      \"thin\": {},\n", w.thin));
    s.push_str(&format!("      \"oracle\": {},\n", json_str(w.oracle)));
    let sweeps = (w.warmup + w.draws * w.thin.max(1)) as u64;
    s.push_str("      \"ferrotherm\": ");
    s.push_str(&side_json(ft_trials, false, sweeps));
    s.push_str(",\n      \"thrml\": ");
    match thrml {
        Some(t) => s.push_str(&side_json(t, true, sweeps)),
        None => s.push_str("null"),
    }
    s.push_str(",\n      \"thrml_main_2d96f4b4\": ");
    match thrml_b {
        Some(t) => s.push_str(&side_json(t, true, sweeps)),
        None => s.push_str("null"),
    }
    s.push_str(",\n      \"parity\": [");
    for (i, (side, ok, reason)) in parity.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!(
            "{{\"side\": {}, \"ok\": {}, \"reason\": {}}}",
            json_str(side),
            if *ok { "true" } else { "false" },
            if reason.is_empty() {
                "null".into()
            } else {
                json_str(reason)
            },
        ));
    }
    s.push_str("],\n      \"mixing\": ");
    s.push_str(&mixing_json(w, ft_trials, thrml, thrml_b, parity));
    s.push_str("\n    }");
    s
}

fn mixing_json(
    w: &Workload,
    ft: &[Chain],
    thrml: Option<&[Chain]>,
    thrml_b: Option<&[Chain]>,
    parity: &[(String, bool, String)],
) -> String {
    let prod = (w.draws * w.thin.max(1)) as f64;
    let (ft_sw, ft_s) = mean_ess(ft, prod);
    let limit_a = limit_kind(parity, "thrml");
    let limit_b = limit_kind(parity, spec::THRML_B_SIDE);
    let (a_sw, a_s) = match limit_a {
        Some(_) => (None, None),
        None => thrml.map_or((None, None), |t| mean_ess(t, prod)),
    };
    let (b_sw, b_s) = match limit_b {
        Some(_) => (None, None),
        None => thrml_b.map_or((None, None), |t| mean_ess(t, prod)),
    };
    format!(
        "{{\n        \"note\": {},\n        \"ferrotherm\": {{\"ess_per_sweep\": {}, \"ess_per_second\": {}, \"wall_ns_per_sweep\": {}}},\n        \
         \"thrml\": {},\n        \"thrml_main_2d96f4b4\": {},\n        \
         \"ess_per_sweep_ratio_ft_over_a\": {},\n        \"ess_per_second_ratio_ft_over_a\": {},\n        \
         \"ess_per_sweep_ratio_ft_over_b\": {},\n        \"ess_per_second_ratio_ft_over_b\": {}\n      }}",
        json_str(
            "ess_per_sweep is mixing per production sweep, ess/(draws*thin). ess_per_second divides by wall seconds. wall_ns_per_sweep is chain seconds, which exclude jit_seconds, times 1e9, divided by warmup+draws*thin. A null ratio is not a ferrotherm win."
        ),
        fmt_num(ft_sw),
        fmt_num(ft_s),
        fmt_num(mean_wall(ft, (w.warmup + w.draws * w.thin.max(1)) as u64)),
        side_mixing(limit_a, a_sw, a_s, mean_wall(thrml.unwrap_or(&[]), (w.warmup + w.draws * w.thin.max(1)) as u64), thrml.is_some()),
        side_mixing(limit_b, b_sw, b_s, mean_wall(thrml_b.unwrap_or(&[]), (w.warmup + w.draws * w.thin.max(1)) as u64), thrml_b.is_some()),
        ratio(ft_sw, a_sw),
        ratio(ft_s, a_s),
        ratio(ft_sw, b_sw),
        ratio(ft_s, b_s),
    )
}

fn side_mixing(
    limit: Option<&str>,
    per_sweep: Option<f64>,
    per_second: Option<f64>,
    wall_ns: Option<f64>,
    present: bool,
) -> String {
    if let Some(kind) = limit {
        return format!("{{\"limit\": {}}}", json_str(kind));
    }
    if !present {
        return "null".into();
    }
    format!(
        "{{\"ess_per_sweep\": {}, \"ess_per_second\": {}, \"wall_ns_per_sweep\": {}}}",
        fmt_num(per_sweep),
        fmt_num(per_second),
        fmt_num(wall_ns)
    )
}

/// Chain seconds exclude `jit_seconds`. The divisor is warmup plus production sweeps.
fn wall_ns_per_sweep(seconds: f64, sweeps: u64) -> Option<f64> {
    if sweeps == 0 || !seconds.is_finite() || seconds < 0.0 {
        None
    } else {
        Some(seconds * 1.0e9 / sweeps as f64)
    }
}

fn mean_wall(trials: &[Chain], sweeps: u64) -> Option<f64> {
    let mut sum = 0.0;
    let mut n = 0.0;
    for c in trials {
        if let Some(v) = wall_ns_per_sweep(c.score.seconds, sweeps) {
            sum += v;
            n += 1.0;
        }
    }
    if n > 0.0 { Some(sum / n) } else { None }
}

fn limit_kind<'a>(parity: &'a [(String, bool, String)], side: &str) -> Option<&'a str> {
    parity.iter().find_map(|(name, _, reason)| {
        if name == side && reason.starts_with("THRML_LIMIT") {
            reason
                .strip_prefix("THRML_LIMIT ")
                .and_then(|rest| rest.split(':').next())
                .map(str::trim)
        } else {
            None
        }
    })
}

fn mean_ess(trials: &[Chain], production_sweeps: f64) -> (Option<f64>, Option<f64>) {
    let mut sw = 0.0;
    let mut n_sw = 0.0;
    let mut sec = 0.0;
    let mut n_sec = 0.0;
    for c in trials {
        if production_sweeps > 0.0 && c.score.ess_energy.is_finite() {
            sw += c.score.ess_energy / production_sweeps;
            n_sw += 1.0;
        }
        if let Some(v) = c.score.ess_per_second.filter(|v| v.is_finite()) {
            sec += v;
            n_sec += 1.0;
        }
    }
    (
        if n_sw > 0.0 { Some(sw / n_sw) } else { None },
        if n_sec > 0.0 { Some(sec / n_sec) } else { None },
    )
}

fn fmt_num(v: Option<f64>) -> String {
    match v {
        Some(v) if v.is_finite() => format!("{v:.8e}"),
        _ => "null".into(),
    }
}

fn ratio(num: Option<f64>, den: Option<f64>) -> String {
    match (num, den) {
        (Some(n), Some(d)) if n.is_finite() && d.is_finite() && d != 0.0 => {
            format!("{:.8e}", n / d)
        }
        _ => "null".into(),
    }
}

fn side_json(trials: &[Chain], external: bool, sweeps: u64) -> String {
    let scores: Vec<&Score> = trials.iter().map(|c| &c.score).collect();
    let mut s = String::from("{\n");
    s.push_str("        \"summary\": ");
    s.push_str(&summary_json(&scores));
    s.push_str(",\n        \"trials\": [\n");
    for (i, c) in trials.iter().enumerate() {
        s.push_str(&trial_json(c, external, sweeps));
        if i + 1 != trials.len() {
            s.push_str(",\n");
        } else {
            s.push('\n');
        }
    }
    s.push_str("        ]\n      }");
    s
}

fn summary_json(scores: &[&Score]) -> String {
    let tv: Vec<f64> = scores.iter().map(|s| s.tv).collect();
    let kl: Vec<f64> = scores.iter().map(|s| s.kl_empirical_exact).collect();
    let tau: Vec<f64> = scores.iter().map(|s| s.tau_int_energy).collect();
    let ess_s: Vec<f64> = scores
        .iter()
        .map(|s| s.ess_per_second.unwrap_or(f64::NAN))
        .collect();
    let de: Vec<f64> = scores.iter().map(|s| s.energy_abs_err).collect();
    let mag: Vec<f64> = scores
        .iter()
        .map(|s| s.magnetization_mae.unwrap_or(f64::NAN))
        .collect();
    format!(
        "{{\n          \"tv\": {},\n          \"kl_empirical_exact\": {},\n          \
         \"tau_int_energy\": {},\n          \"ess_per_second\": {},\n          \
         \"energy_abs_err\": {},\n          \"magnetization_mae\": {}\n        }}",
        ci_json(&tv),
        ci_json(&kl),
        ci_json(&tau),
        ci_json(&ess_s),
        ci_json(&de),
        ci_json(&mag),
    )
}

fn ci_json(xs: &[f64]) -> String {
    match mean_ci_95(xs) {
        Some((mean, lo, hi, n)) => {
            format!("{{\"mean\": {mean:.8e}, \"lo95\": {lo:.8e}, \"hi95\": {hi:.8e}, \"n\": {n}}}")
        }
        None => "null".into(),
    }
}

fn trial_json(c: &Chain, external: bool, sweeps: u64) -> String {
    let sc = &c.score;
    let kl_rev = match sc.kl_exact_empirical {
        Some(v) => format!("{v:.8e}"),
        None => "null".into(),
    };
    let ess_s = match sc.ess_per_second {
        Some(v) => format!("{v:.8e}"),
        None => "null".into(),
    };
    let mag = match sc.magnetization_mae {
        Some(v) => format!("{v:.8e}"),
        None => "null".into(),
    };
    let jit_flag = if external { "true" } else { "false" };
    let jit_s = fmt_f64(c.jit_seconds);
    format!(
        "          {{\"draws\": {}, \"seconds\": {:.8e}, \"jit_seconds\": {jit_s}, \
         \"jit_excluded_from_seconds\": {jit_flag}, \
         \"tv\": {}, \"kl_empirical_exact\": {}, \"kl_exact_empirical\": {kl_rev}, \
         \"missed_states\": {}, \"tau_int_energy\": {:.8e}, \"ess_energy\": {:.8e}, \
         \"ess_per_second\": {ess_s}, \"ess_per_sweep\": {}, \"wall_ns_per_sweep\": {}, \
         \"submits\": {}, \"sweeps_per_submit\": {}, \"mean_energy\": {}, \
         \"exact_mean_energy\": {}, \"energy_abs_err\": {}, \"magnetization_mae\": {mag}, \
         \"gibbs_updates\": {}, \"reads\": {}, \
         \"measured_joules_per_ess\": null}}",
        sc.draws,
        sc.seconds,
        fmt_finite(sc.tv),
        fmt_finite(sc.kl_empirical_exact),
        sc.missed_states,
        sc.tau_int_energy,
        sc.ess_energy,
        if sc.draws > 0 && sc.ess_energy.is_finite() {
            fmt_finite(sc.ess_energy / sc.draws as f64)
        } else {
            "null".into()
        },
        fmt_num(wall_ns_per_sweep(sc.seconds, sweeps)),
        if c.submits > 0 { c.submits.to_string() } else { "null".into() },
        if c.submits > 0 { fmt_num(Some(c.sweeps_per_submit)) } else { "null".into() },
        fmt_finite(sc.mean_energy),
        fmt_finite(sc.exact_mean_energy),
        fmt_finite(sc.energy_abs_err),
        c.updates,
        c.reads,
    )
}

struct WindowRecord {
    load1: Vec<f64>,
    signal: Vec<f64>,
    ps: String,
}

fn fmt_series(values: &[f64]) -> String {
    values
        .iter()
        .map(|v| {
            if v.is_finite() {
                format!("{v:.4}")
            } else {
                "null".into()
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn read_ps_cpu() -> Vec<(u32, u32, f64)> {
    let Ok(out) = Command::new("ps")
        .args(["-axo", "pid=,ppid=,time="])
        .output()
    else {
        return Vec::new();
    };
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(energy::parse_ps_cpu_row)
        .collect()
}

fn attribution_ps() -> String {
    let rows = read_ps_cpu();
    let self_pid = std::process::id();
    let mut parent = std::collections::BTreeMap::new();
    for (pid, ppid, _) in &rows {
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
    let mut foreign: Vec<(u32, u32, f64)> = rows
        .iter()
        .copied()
        .filter(|(pid, _, _)| !harness(*pid))
        .collect();
    foreign.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    foreign.truncate(40);
    foreign
        .iter()
        .map(|(pid, ppid, cpu)| format!("{pid} {ppid} {cpu:.3}"))
        .collect::<Vec<_>>()
        .join("; ")
}

fn sample_window<T>(pinned: &[usize], body: impl FnOnce() -> T) -> (T, WindowRecord) {
    let pinned = pinned.to_vec();
    let loads = std::sync::Mutex::new(Vec::<f64>::new());
    let signals = std::sync::Mutex::new(Vec::<f64>::new());
    let stop = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|scope| {
        let handle = scope.spawn(|| {
            let mut prev_stat = fs::read_to_string("/proc/stat").unwrap_or_default();
            let mut prev_cpu = if cfg!(target_os = "macos") {
                read_ps_cpu()
            } else {
                Vec::new()
            };
            let mut prev_at = std::time::Instant::now();
            let mut seen = false;
            loop {
                let load = ferrotherm::host::load_average().unwrap_or(f64::NAN);
                loads.lock().unwrap().push(load);
                if cfg!(target_os = "linux") {
                    let now = fs::read_to_string("/proc/stat").unwrap_or_default();
                    if seen
                        && let Some(fraction) =
                            energy::other_core_fraction(&prev_stat, &now, &pinned)
                    {
                        signals.lock().unwrap().push(fraction);
                    }
                    prev_stat = now;
                } else if cfg!(target_os = "macos") {
                    let now = read_ps_cpu();
                    let wall = prev_at.elapsed().as_secs_f64();
                    if seen
                        && let Some(fraction) =
                            energy::foreign_cpu_fraction(&prev_cpu, &now, std::process::id(), wall)
                    {
                        signals.lock().unwrap().push(fraction);
                    }
                    prev_cpu = now;
                    prev_at = std::time::Instant::now();
                }
                seen = true;
                if stop.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
                for _ in 0..10 {
                    if stop.load(std::sync::atomic::Ordering::Relaxed) {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
            }
        });
        let value = body();
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        let _ = handle.join();
        let record = WindowRecord {
            load1: std::mem::take(&mut *loads.lock().unwrap()),
            signal: std::mem::take(&mut *signals.lock().unwrap()),
            ps: attribution_ps(),
        };
        (value, record)
    })
}

fn settle_for_bracket(contended: bool) -> Result<Vec<f64>, String> {
    if contended {
        return Ok(Vec::new());
    }
    let mut samples = Vec::new();
    loop {
        samples.push(ferrotherm::host::load_average());
        let outcome = energy::settle_outcome(&samples, energy::SETTLE_STREAK, energy::SETTLE_BOUND);
        if outcome.settled {
            return Ok(outcome.samples);
        }
        if samples.len() >= energy::SETTLE_BOUND {
            return Err(format!(
                "idle bracket refused: {}. The threshold is unchanged at 2.",
                outcome.reason
            ));
        }
        std::thread::sleep(energy::ENERGY_WATCH_INTERVAL);
    }
}

fn watched_bracket(probe: &mut Probe, contended: bool) -> Result<BracketSample, String> {
    let (result, watch) = sample_window(&[], || sample_bracket(probe, contended));
    let bracket = result?;
    if !contended {
        let opts: Vec<Option<f64>> = watch.load1.iter().copied().map(Some).collect();
        if let Some(reason) = bracket_refusal("idle bracket", &opts) {
            return Err(reason);
        }
    }
    Ok(bracket)
}

fn bracket_refusal(which: &str, samples: &[Option<f64>]) -> Option<String> {
    let offender = energy::bracket_load_offender(samples);
    let verdict = energy::grade_energy_watch(samples);
    if offender.is_none() && verdict.evidence_grade == energy::GRADE_QUIET {
        return None;
    }
    let bad = offender
        .or_else(|| verdict.abort.map(|reading| reading.load1))
        .unwrap_or(f64::NAN);
    Some(format!(
        "{which} refused: a 1 Hz load1 sample was {bad}, above 2. The threshold is unchanged."
    ))
}

fn energy_section(
    probe: &mut Probe,
    work: &[Workload],
    contended: bool,
    trials: usize,
    python: &Path,
    python_b: &Path,
    runner: &Path,
    spec: &Path,
    thrml_ok: bool,
    device: &str,
    pinned: &[usize],
    preflight_held: bool,
    rule: &energy::TrialWindowRule,
    grid: Option<ft::GridChoice>,
    gpu: bool,
) -> (String, &'static str) {
    let poll = if contended {
        poll_quiet(&[ferrotherm::host::load_average()])
    } else if preflight_held {
        // The 60 s preflight just held. Settle, then the idle baseline. No THRML import here.
        let load = ferrotherm::host::load_average();
        energy::QuietPoll {
            outcome: "quiet",
            samples: load.into_iter().collect(),
            accepted: load,
            threshold: ferrotherm::host::QUIET_LOAD,
        }
    } else {
        wait_for_quiet_host()
    };
    let quiet = poll.outcome == "quiet";
    let mut grade = grade_for(contended, quiet);
    let has_instrument = probe.rapl.is_some()
        || probe.perf.is_some()
        || probe.wall.is_some()
        || probe.nvml.is_some()
        || probe.ina238.is_some()
        || probe.tegra.is_some()
        || !probe.subrails.is_empty();
    if grade == "load_gated" {
        let gated = energy::load_gated(
            "No joules. The quiet-host gate is unchanged (load average above 2). After the bounded \
             wait the host was still contended, and this run did not opt into contended-host mode, \
             so no energy figure was taken. A contended figure would be a different grade and is \
             not implied here.",
        );
        return (
            energy_json(
                probe,
                &poll,
                grade,
                gated.evidence,
                None,
                &[],
                &gated.note,
                rule,
                &[],
            ),
            grade,
        );
    }
    if trials < 5 {
        return (
            energy_json(
                probe,
                &poll,
                grade,
                "too_few_trials",
                None,
                &[],
                "No joules. The energy head-to-head needs at least 5 interleaved trials so a 95% interval exists.",
                rule,
                &[],
            ),
            grade,
        );
    }
    if !has_instrument {
        let missing = Joules::unavailable(
            0.0,
            "none",
            "No joules were obtained. No macmon, INA3221 sub-rail, INA238 VIN, tegrastats VIN or \
             VDD_IN, RAPL powercap, perf power/energy-pkg, or NVML counter opened on this host. \
             Nothing was filled in from a device-price table.",
        );
        return (
            energy_json(
                probe,
                &poll,
                grade,
                missing.evidence,
                None,
                &[],
                &missing.note,
                rule,
                &[],
            ),
            grade,
        );
    }
    if !contended && energy::linux_quiet_needs_pin(std::env::consts::OS, pinned.len()) {
        let note = "energy-phase quiet grade refused: on Linux --cpus is required. The load threshold is unchanged.";
        return (
            energy_json(
                probe,
                &poll,
                "load_gated",
                "load_gated",
                None,
                &[],
                note,
                rule,
                &[],
            ),
            "load_gated",
        );
    }
    // THRML is not started until the first arm sample, inside that trial window.
    // A and B are separate pairings. Their ferrotherm rows do not share a summary.
    let _ = (&grid, gpu);
    let mut slot_a = ThrmlSlot {
        session: None,
        side: spec::THRML_A_SIDE,
        baseline: "A",
        python: python.to_path_buf(),
        runner: runner.to_path_buf(),
        spec: spec.to_path_buf(),
        device: device.to_string(),
        enabled: thrml_ok,
        ex_jit_ns: Vec::new(),
        oom: false,
    };
    let mut slot_b = ThrmlSlot {
        session: None,
        side: spec::THRML_B_SIDE,
        baseline: "B",
        python: python_b.to_path_buf(),
        runner: runner.to_path_buf(),
        spec: spec.to_path_buf(),
        device: device.to_string(),
        enabled: thrml_ok && !python_b.as_os_str().is_empty(),
        ex_jit_ns: Vec::new(),
        oom: false,
    };
    let baseline_settle = match settle_for_bracket(contended) {
        Ok(samples) => samples,
        Err(reason) => {
            return (
                energy_json(
                    probe,
                    &poll,
                    "load_gated",
                    "load_gated",
                    None,
                    &[],
                    &reason,
                    rule,
                    &[],
                ),
                "load_gated",
            );
        }
    };
    let (idle_res, idle_watch) = sample_window(&[], || take_idle(probe, contended));
    let idle = match idle_res {
        Ok(b) => b,
        Err(e) => {
            let grade = if contended {
                GRADE_CONTENDED
            } else {
                "load_gated"
            };
            return (
                energy_json(
                    probe,
                    &poll,
                    grade,
                    "load_gated",
                    None,
                    &[],
                    &e,
                    rule,
                    &baseline_settle,
                ),
                grade,
            );
        }
    };
    if !contended {
        let opts: Vec<Option<f64>> = idle_watch.load1.iter().copied().map(Some).collect();
        if let Some(reason) = bracket_refusal("idle baseline", &opts) {
            return (
                energy_json(
                    probe,
                    &poll,
                    "load_gated",
                    "load_gated",
                    None,
                    &[],
                    &reason,
                    rule,
                    &baseline_settle,
                ),
                "load_gated",
            );
        }
    }
    let mut window_rule = *rule;
    if contended {
        window_rule.enforce = false;
    }
    let stat_before_run = fs::read_to_string("/proc/stat").unwrap_or_default();
    let mut rows = Vec::new();
    let mut phase_refused = String::new();
    'pairs: for w in work {
        if w.open_grid().is_some() {
            // One THRML process per side. A CUDA OOM on 4096 must not be an unlabeled
            // death of the process that also held the smaller sides, and it must not
            // be scored as a ferrotherm win.
            let stem = spec.file_stem().and_then(|s| s.to_str()).unwrap_or("spec");
            let dir = spec.parent().unwrap_or_else(|| Path::new("."));
            let one = dir.join(format!("{stem}.{}.spec", w.id));
            if fs::write(&one, spec::write_spec(std::slice::from_ref(w))).is_ok() {
                slot_a.restart(&one);
                slot_b.restart(&one);
            }
        }
        for pass in 0..2 {
            let (pairing, other_side, slot) = match pass {
                0 => ("A", spec::THRML_A_SIDE, &mut slot_a),
                _ => ("B", spec::THRML_B_SIDE, &mut slot_b),
            };
            if pass == 1 && python_b.as_os_str().is_empty() {
                // No B interpreter. Do not time an empty window and call it B's joules.
                continue;
            }
            for pair in 0..trials {
                let settle_samples = match settle_for_bracket(contended) {
                    Ok(samples) => samples,
                    Err(reason) => {
                        phase_refused = reason;
                        break 'pairs;
                    }
                };
                let settle_txt = fmt_series(&settle_samples);
                let before_b = match watched_bracket(probe, contended) {
                    Ok(v) => v,
                    Err(e) => {
                        phase_refused = e;
                        break 'pairs;
                    }
                };
                let mut got: Vec<(
                    &'static str,
                    Vec<SideMeasure>,
                    energy::CpuSnap,
                    energy::CpuSnap,
                )> = Vec::new();
                let order = if pass == 0 {
                    energy::pair_order(pair)
                } else {
                    energy::pair_order_against(pair, other_side)
                };
                for side in order {
                    let seed = w.seed0 + pair as u64 + 10_000;
                    let before = read_cpu_snap();
                    let mut measured = measure_side(
                        probe,
                        side,
                        w,
                        seed,
                        &mut *slot,
                        device,
                        pinned,
                        &window_rule,
                    );
                    if measured.first().map(|m| m.instrument) != Some("nvml")
                        && probe.nvml.is_some()
                    {
                        measured.extend(measure_counter(
                            probe,
                            side,
                            w,
                            seed,
                            &mut *slot,
                            CounterKind::Nvml,
                        ));
                    }
                    let after = read_cpu_snap();
                    got.push((side, measured, before, after));
                }
                let after_b = match watched_bracket(probe, contended) {
                    Ok(v) => v,
                    Err(e) => {
                        phase_refused = e;
                        break 'pairs;
                    }
                };
                let (mean_w, sigma, _drift, drift_ok) = energy::combine_bracket(
                    before_b.watts,
                    before_b.sigma,
                    after_b.watts,
                    after_b.sigma,
                );
                if !contended && energy::macos_drift_refuses_quiet(std::env::consts::OS, drift_ok) {
                    phase_refused = "bracket drift exceeded 3 sigma".into();
                }
                let (core_mean, core_sigma, core_drift, core_ok) = match (
                    before_b.core_watts,
                    before_b.core_sigma,
                    after_b.core_watts,
                    after_b.core_sigma,
                ) {
                    (Some(a), Some(asg), Some(b), Some(bsg)) => {
                        energy::combine_bracket(a, asg, b, bsg)
                    }
                    _ => (f64::NAN, f64::NAN, f64::NAN, false),
                };
                let stat_after = fs::read_to_string("/proc/stat").unwrap_or_default();
                let other = energy::other_core_busy(&stat_before_run, &stat_after, pinned);
                let other_txt = other
                    .iter()
                    .map(|(c, f)| format!("{c}:{f:.4}"))
                    .collect::<Vec<_>>()
                    .join(",");
                if w.open_grid().is_some() {
                    let seed0 = w.seed0 + pair as u64 + 10_000;
                    for (side, measures, _, _) in &mut got {
                        let reps = measures.first().map(|m| m.repetitions).unwrap_or(0);
                        let ess = remix_ess(side, w, seed0, reps, slot);
                        for m in measures.iter_mut() {
                            m.ess = ess;
                        }
                    }
                }
                for (side, measures, before, after) in &got {
                    for measured in measures {
                        let rank = energy_rank(measured.instrument, &probe.hostname);
                        let core_row = measured.scope == "core";
                        let (row_idle, row_sigma, row_drift_ok) = if core_row {
                            (core_mean, core_sigma, core_ok)
                        } else if let Some((before_w, before_s, after_w, after_s)) =
                            subrail_bracket(&before_b, &after_b, &measured.scope)
                        {
                            let (mean, sig, _, ok) =
                                energy::combine_bracket(before_w, before_s, after_w, after_s);
                            (mean, sig, ok)
                        } else {
                            (mean_w, sigma, drift_ok)
                        };
                        let joules = trial_joules(
                            measured.gross,
                            row_idle,
                            row_sigma,
                            measured.seconds,
                            measured.instrument,
                            &measured.scope,
                            grade,
                            measured.counter_updates,
                            row_drift_ok,
                        );
                        let mut row = EnergyTrial::from_parts(
                            w.id,
                            side,
                            pair,
                            measured.repetitions,
                            measured.counter_updates,
                            measured.window_seconds,
                            joules,
                            measured.ess,
                            after.load1,
                            snap_other_cpu(before, after),
                            grade,
                            rank,
                            pairing,
                            fabric_name(side, pairing, device, gpu),
                        );
                        row.idle_before_w = Some(if core_row {
                            before_b.core_watts.unwrap_or(f64::NAN)
                        } else {
                            subrail_level(&before_b, &measured.scope).unwrap_or(before_b.watts)
                        });
                        row.idle_after_w = Some(if core_row {
                            after_b.core_watts.unwrap_or(f64::NAN)
                        } else {
                            subrail_level(&after_b, &measured.scope).unwrap_or(after_b.watts)
                        });
                        row.baseline_drift_w = Some(if core_row {
                            core_drift
                        } else {
                            (row.idle_before_w.unwrap_or(0.0) - row.idle_after_w.unwrap_or(0.0))
                                .abs()
                        });
                        row.other_core_busy = other_txt.clone();
                        row.cadence_s = measured.cadence_s;
                        row.left_edge = measured.left_edge.to_string();
                        row.right_edge = measured.right_edge.to_string();
                        row.uncovered_s = measured.uncovered_s;
                        row.samples = measured.samples;
                        row.quantization_window_j = measured.quantization_j;
                        row.cadence_edge_window_j = measured.cadence_edge_j;
                        row.window_load1 = measured.window_load1.clone();
                        row.window_signal = measured.window_signal.clone();
                        row.ps_attribution = measured.ps_attribution.clone();
                        row.trial_refusal = measured.trial_refusal.clone();
                        row.settle_load1 = settle_txt.clone();
                        if w.open_grid().is_some() {
                            let sweeps = (w.warmup + w.draws * w.thin.max(1)) as u64;
                            let produced = (w.draws * w.thin.max(1)) as f64;
                            row.sweeps_per_rep = sweeps;
                            row.ess_per_sweep = if produced > 0.0 && row.ess.is_finite() {
                                row.ess / produced
                            } else {
                                f64::NAN
                            };
                            row.seconds_per_sweep = if sweeps > 0 {
                                row.joules.seconds / sweeps as f64
                            } else {
                                f64::NAN
                            };
                            row.joules_per_sweep = row
                                .joules
                                .incremental
                                .filter(|j| sweeps > 0 && j.is_finite())
                                .map(|j| j / sweeps as f64);
                            if *side == "ferrotherm" {
                                row.wall_ns_per_sweep = if sweeps > 0 {
                                    row.joules.seconds * 1.0e9 / sweeps as f64
                                } else {
                                    f64::NAN
                                };
                            } else if slot.oom {
                                let limit = thrml_limit_reason("oom");
                                row.trial_refusal = if row.trial_refusal.is_empty() {
                                    limit
                                } else {
                                    format!("{limit} {}", row.trial_refusal)
                                };
                                row.wall_ns_per_sweep = f64::NAN;
                            } else {
                                let n = slot.ex_jit_ns.len();
                                row.wall_ns_per_sweep = if n == 0 {
                                    f64::NAN
                                } else {
                                    slot.ex_jit_ns.iter().sum::<f64>() / n as f64
                                };
                            }
                        }
                        if !measured.trial_refusal.is_empty() {
                            row.joules.attributable = false;
                            row.joules.evidence = "load_gated";
                            row.joules.note = measured.trial_refusal.clone();
                            row.joules_per_ess = None;
                            if phase_refused.is_empty() {
                                phase_refused = measured.trial_refusal.clone();
                            }
                        }
                        rows.push(row);
                    }
                }
                stamp_pairs(
                    &mut rows, w.id, pair, pairing, sigma, drift_ok, core_sigma, core_ok,
                );
                if !phase_refused.is_empty() {
                    break 'pairs;
                }
            }
        }
    }
    if !phase_refused.is_empty() && !contended {
        grade = "load_gated";
    }
    let mut note = if phase_refused.is_empty() {
        energy_note(grade)
    } else {
        phase_refused
    };
    if device == "cuda" && probe.nvml_note.contains("NOT_SUPPORTED") {
        note.push(' ');
        note.push_str(&probe.nvml_note);
    }
    note.push_str(
        " THRML, when it runs, is started inside the first THRML trial window. Its CPU is the B arm, not foreign load, and it is not subtracted from joules.",
    );
    let outcome = if phase_refused_left(&note, grade) {
        "load_gated"
    } else {
        "measured"
    };
    (
        energy_json(
            probe,
            &poll,
            grade,
            outcome,
            Some(idle),
            &rows,
            &note,
            rule,
            &baseline_settle,
        ),
        grade,
    )
}

fn phase_refused_left(note: &str, grade: &str) -> bool {
    grade == "load_gated" && !note.starts_with("Evidence grade")
}

fn energy_note(grade: &str) -> String {
    if grade == GRADE_CONTENDED {
        "Evidence grade contended host. Each pair is ferrotherm then THRML, or the reverse, \
         with an idle bracket before and after the pair. Attribution refuses a negative \
         incremental, a below-noise incremental, and a bracket that drifted. The paired \
         difference keeps its sign. RAPL package and RAPL core are separate scopes. These \
         figures are not quiet-host evidence. The quiet gate is unchanged."
            .into()
    } else {
        "Evidence grade quiet. The host cleared the unchanged load gate (at or under 2). \
         Each trial repeats the sampler for at least 2 s and reports joules per repetition. \
         Attribution needs 8 counter updates. Summaries are per workload. RAPL package, \
         RAPL core, and macmon SoC telemetry are not combined. RAPL on framework-128gb-0000 \
         reads package energy_uj without root and is primary; macmon sys_power is secondary \
         and is not wall power."
            .into()
    }
}

struct BracketSample {
    watts: f64,
    sigma: f64,
    core_watts: Option<f64>,
    core_sigma: Option<f64>,
    /// `(scope, watts, sigma)` for each INA3221 sub-rail. Not summed.
    subrails: Vec<(String, f64, f64)>,
}

fn subrail_level(bracket: &BracketSample, scope: &str) -> Option<f64> {
    bracket
        .subrails
        .iter()
        .find(|(name, _, _)| name == scope)
        .map(|(_, watts, _)| *watts)
}

fn subrail_bracket(
    before: &BracketSample,
    after: &BracketSample,
    scope: &str,
) -> Option<(f64, f64, f64, f64)> {
    if !scope.starts_with("subrail:") {
        return None;
    }
    let (_, bw, bs) = before.subrails.iter().find(|(name, _, _)| name == scope)?;
    let (_, aw, asg) = after.subrails.iter().find(|(name, _, _)| name == scope)?;
    Some((*bw, *bs, *aw, *asg))
}

fn sample_bracket(probe: &mut Probe, contended: bool) -> Result<BracketSample, String> {
    let t0 = std::time::Instant::now();
    if let Some(wall) = probe.wall.as_mut() {
        let b = if contended {
            wall.observe_idle(std::time::Duration::from_secs(1))?
        } else {
            wall.idle(std::time::Duration::from_secs(1))?
        };
        return Ok(BracketSample {
            watts: b.watts,
            sigma: b.sigma,
            core_watts: None,
            core_sigma: None,
            subrails: subrail_idles(probe, t0),
        });
    }
    if let Some(series) = probe.ina238.as_ref() {
        std::thread::sleep(std::time::Duration::from_secs(1));
        let seconds = t0.elapsed().as_secs_f64();
        std::thread::sleep(std::time::Duration::from_millis(250));
        let (watts, sigma) = series_idle(&series.since_stamped(t0), t0, seconds, series.counter)?;
        return Ok(BracketSample {
            watts,
            sigma,
            core_watts: None,
            core_sigma: None,
            subrails: subrail_idles(probe, t0),
        });
    }
    if probe.tegra.is_some() {
        let t0 = std::time::Instant::now();
        std::thread::sleep(std::time::Duration::from_secs(1));
        let seconds = t0.elapsed().as_secs_f64();
        std::thread::sleep(std::time::Duration::from_millis(250));
        let stamped = probe
            .tegra
            .as_ref()
            .map(|t| t.since_stamped(t0))
            .unwrap_or_default();
        let series = series_secs(t0, &stamped);
        let integ = ferrotherm_meter::trapezoid(&series, 0.0, seconds)?;
        let inside: Vec<f64> = series
            .iter()
            .filter(|(t, _)| *t >= 0.0 && *t <= seconds)
            .map(|(_, w)| *w)
            .collect();
        let (_, sigma) = energy::idle_stats(&inside)
            .ok_or_else(|| "tegrastats bracket produced fewer than two samples".to_string())?;
        if integ.covered_s <= 0.0 {
            return Err("tegrastats bracket covered no time".into());
        }
        return Ok(BracketSample {
            watts: integ.joules / integ.covered_s,
            sigma,
            core_watts: None,
            core_sigma: None,
            subrails: subrail_idles(probe, t0),
        });
    }
    let mut pkg = Vec::new();
    let mut core = Vec::new();
    for _ in 0..4 {
        let (j, cj, secs) = counter_slice_pair(probe, 0.25)?;
        if secs > 0.0 {
            pkg.push(j / secs);
            if let Some(c) = cj {
                core.push(c / secs);
            }
        }
    }
    let (watts, sigma) = energy::idle_stats(&pkg)
        .ok_or_else(|| "counter bracket produced fewer than two windows".to_string())?;
    let core_pair = energy::idle_stats(&core);
    Ok(BracketSample {
        watts,
        sigma,
        core_watts: core_pair.map(|p| p.0),
        core_sigma: core_pair.map(|p| p.1),
        subrails: subrail_idles(probe, t0),
    })
}

fn subrail_idles(probe: &Probe, from: std::time::Instant) -> Vec<(String, f64, f64)> {
    let seconds = from.elapsed().as_secs_f64().max(0.001);
    probe
        .subrails
        .iter()
        .filter_map(|series| {
            let (watts, sigma) =
                series_idle(&series.since_stamped(from), from, seconds, series.counter).ok()?;
            Some((series.scope.clone(), watts, sigma))
        })
        .collect()
}

fn series_idle(
    stamped: &[(std::time::Instant, f64)],
    from: std::time::Instant,
    seconds: f64,
    counter: bool,
) -> Result<(f64, f64), String> {
    if counter {
        let inside: Vec<f64> = stamped
            .iter()
            .filter(|(t, _)| *t >= from)
            .map(|(_, j)| *j)
            .collect();
        if inside.len() < 2 || seconds <= 0.0 {
            return Err("module-input counter bracket produced fewer than two samples".into());
        }
        let watts = (inside[inside.len() - 1] - inside[0]) / seconds;
        let steps: Vec<f64> = inside
            .windows(2)
            .map(|w| (w[1] - w[0]) * inside.len() as f64 / seconds)
            .collect();
        let (_, sigma) = energy::idle_stats(&steps).unwrap_or((watts, 0.0));
        return Ok((watts, sigma));
    }
    let series = series_secs(from, stamped);
    let integ = ferrotherm_meter::trapezoid(&series, 0.0, seconds)?;
    if integ.covered_s <= 0.0 {
        return Err("power bracket covered no time".into());
    }
    let inside: Vec<f64> = series
        .iter()
        .filter(|(t, _)| *t >= 0.0 && *t <= seconds)
        .map(|(_, w)| *w)
        .collect();
    let (_, sigma) = energy::idle_stats(&inside)
        .ok_or_else(|| "power bracket produced fewer than two samples".to_string())?;
    Ok((integ.joules / integ.covered_s, sigma))
}

fn counter_slice_pair(probe: &mut Probe, secs: f64) -> Result<(f64, Option<f64>, f64), String> {
    if let Some(r) = probe.rapl.as_ref() {
        let a = r.read_uj()?;
        let ca = probe.rapl_core.as_ref().and_then(|c| c.read_uj().ok());
        let t = std::time::Instant::now();
        std::thread::sleep(std::time::Duration::from_secs_f64(secs));
        let b = r.read_uj()?;
        let cb = probe.rapl_core.as_ref().and_then(|c| c.read_uj().ok());
        let dt = t.elapsed().as_secs_f64();
        let core = match (ca, cb, probe.rapl_core.as_ref()) {
            (Some(x), Some(y), Some(c)) => Some(c.delta_j(x, y)),
            _ => None,
        };
        return Ok((r.delta_j(a, b), core, dt));
    }
    let (j, dt) = counter_slice(probe, secs)?;
    Ok((j, None, dt))
}

fn stamp_pairs(
    rows: &mut [EnergyTrial],
    workload: &str,
    pair: usize,
    pairing: &str,
    sigma: f64,
    drift_ok: bool,
    core_sigma: f64,
    core_ok: bool,
) {
    let scopes: Vec<String> = rows
        .iter()
        .filter(|r| r.workload == workload && r.trial == pair && r.pairing == pairing)
        .map(|r| r.joules.scope.clone())
        .collect();
    let mut uniq = scopes;
    uniq.sort();
    uniq.dedup();
    for scope in uniq {
        let pick = |side: &str| -> Option<(f64, f64, u32)> {
            rows.iter()
                .find(|r| {
                    r.workload == workload
                        && r.trial == pair
                        && r.pairing == pairing
                        && r.side == side
                        && r.joules.scope == scope
                })
                .and_then(|r| {
                    r.joules
                        .incremental
                        .map(|inc| (inc, r.joules.seconds, r.counter_updates))
                })
        };
        let Some((a, sa, ua)) = pick("ferrotherm") else {
            continue;
        };
        let Some((b, sb, ub)) = pick(if pairing == "B" {
            spec::THRML_B_SIDE
        } else {
            spec::THRML_A_SIDE
        }) else {
            continue;
        };
        let delta = energy::paired_delta(a, b);
        let sig = if scope == "core" { core_sigma } else { sigma };
        let gate = if scope == "core" { core_ok } else { drift_ok };
        let ok = energy::paired_attributable(delta, sig, sa, sb, gate, ua, ub);
        for row in rows.iter_mut() {
            if row.workload == workload
                && row.trial == pair
                && row.pairing == pairing
                && row.joules.scope == scope
            {
                row.paired_delta_j = Some(delta);
                row.paired_attributable = ok;
            }
        }
    }
}

fn take_idle(probe: &mut Probe, contended: bool) -> Result<(f64, f64), String> {
    if let Some(wall) = probe.wall.as_mut() {
        let b = if contended {
            wall.observe_idle(std::time::Duration::from_secs(2))?
        } else {
            energy::idle_baseline(wall)?
        };
        return Ok((b.watts, b.sigma));
    }
    counter_idle(probe)
}

fn counter_idle(probe: &mut Probe) -> Result<(f64, f64), String> {
    let mut windows = Vec::new();
    for _ in 0..6 {
        let (j, secs) = counter_slice(probe, 0.25)?;
        windows.push((j, secs));
    }
    energy::watts_from_windows(&windows)
        .ok_or_else(|| "idle counter produced fewer than two windows".into())
}

fn counter_slice(probe: &mut Probe, secs: f64) -> Result<(f64, f64), String> {
    if let Some(r) = probe.rapl.as_ref() {
        let a = r.read_uj()?;
        let t = Instant::now();
        std::thread::sleep(std::time::Duration::from_secs_f64(secs));
        let b = r.read_uj()?;
        return Ok((r.delta_j(a, b), t.elapsed().as_secs_f64()));
    }
    if let Some(p) = probe.perf.as_ref() {
        let a = p.read_j()?;
        let t = Instant::now();
        std::thread::sleep(std::time::Duration::from_secs_f64(secs));
        let b = p.read_j()?;
        let dt = t.elapsed().as_secs_f64();
        let gross = if b >= a { b - a } else { 0.0 };
        return Ok((gross, dt));
    }
    if let Some(n) = probe.nvml.as_ref() {
        let a = n.joules()?;
        let t = Instant::now();
        std::thread::sleep(std::time::Duration::from_secs_f64(secs));
        let b = n.joules()?;
        return Ok((
            energy::counter_delta_j(a, b, u64::MAX as f64 / 1000.0),
            t.elapsed().as_secs_f64(),
        ));
    }
    Err("no counter".into())
}

struct SideMeasure {
    ess: f64,
    gross: f64,
    seconds: f64,
    window_seconds: f64,
    repetitions: u32,
    counter_updates: u32,
    instrument: &'static str,
    scope: String,
    cadence_s: f64,
    left_edge: &'static str,
    right_edge: &'static str,
    uncovered_s: f64,
    samples: u32,
    quantization_j: Option<f64>,
    cadence_edge_j: f64,
    window_load1: String,
    window_signal: String,
    ps_attribution: String,
    trial_refusal: String,
}

fn measure_side(
    probe: &mut Probe,
    side: &str,
    w: &Workload,
    seed: u64,
    slot: &mut ThrmlSlot,
    device: &str,
    pinned: &[usize],
    rule: &energy::TrialWindowRule,
) -> Vec<SideMeasure> {
    if side != "ferrotherm" {
        slot.ex_jit_ns.clear();
        slot.oom = false;
        EXJIT_NS.with(|v| v.borrow_mut().clear());
        EXJIT_OOM.with(|c| c.set(false));
    }
    let t0 = std::time::Instant::now();
    let cuda_vin =
        device == "cuda" && probe.nvml.is_none() && probe.nvml_note.contains("NOT_SUPPORTED");
    let (mut out, watch) = sample_window(pinned, || {
        if cuda_vin && let Some(series) = probe.ina238.as_ref() {
            vec![measure_hwmon(series, side, w, seed, slot)]
        } else if cuda_vin && probe.tegra.is_some() {
            vec![measure_tegra(probe, side, w, seed, slot)]
        } else if probe.wall.is_some() && !cuda_vin {
            vec![measure_wall(probe, side, w, seed, slot)]
        } else if let Some(series) = probe.ina238.as_ref() {
            vec![measure_hwmon(series, side, w, seed, slot)]
        } else if probe.tegra.is_some() {
            vec![measure_tegra(probe, side, w, seed, slot)]
        } else if probe.wall.is_some() {
            vec![measure_wall(probe, side, w, seed, slot)]
        } else if probe.rapl.is_some() {
            measure_counter(probe, side, w, seed, slot, CounterKind::Rapl)
        } else if probe.perf.is_some() {
            measure_counter(probe, side, w, seed, slot, CounterKind::Perf)
        } else if probe.nvml.is_some() {
            measure_counter(probe, side, w, seed, slot, CounterKind::Nvml)
        } else {
            Vec::new()
        }
    });
    let (mean_max, sample_max, apply) = if cfg!(target_os = "macos") {
        (rule.foreign_cpu_mean_max, rule.foreign_cpu_sample_max, true)
    } else if cfg!(target_os = "linux") {
        (rule.other_core_mean_max, rule.other_core_sample_max, true)
    } else {
        (f64::INFINITY, f64::INFINITY, false)
    };
    let refusal = if rule.enforce && apply {
        energy::fraction_refusal(&watch.signal, mean_max, sample_max)
            .unwrap_or("")
            .to_string()
    } else {
        String::new()
    };
    let (reps, ess_sum, window_s) = match out.first() {
        Some(row) if row.repetitions > 0 => (
            row.repetitions,
            row.ess * f64::from(row.repetitions),
            row.window_seconds,
        ),
        _ => (0, f64::NAN, 0.0),
    };
    if reps > 0 && !probe.subrails.is_empty() {
        std::thread::sleep(std::time::Duration::from_millis(250));
        for series in &probe.subrails {
            out.push(integrate_logged(series, t0, reps, ess_sum, window_s));
        }
    }
    if side != "ferrotherm" {
        slot.ex_jit_ns = EXJIT_NS.with(|v| v.borrow().clone());
        // arm() may have set oom when the process died before READY. cost() sets the thread-local.
        slot.oom = slot.oom || EXJIT_OOM.with(|c| c.get());
    }
    let load_txt = fmt_series(&watch.load1);
    let signal_txt = fmt_series(&watch.signal);
    for row in &mut out {
        row.window_load1 = load_txt.clone();
        row.window_signal = signal_txt.clone();
        row.ps_attribution = watch.ps.clone();
        row.trial_refusal = refusal.clone();
    }
    out
}

fn measure_hwmon(
    series: &energy::HwmonSeries,
    side: &str,
    w: &Workload,
    seed: u64,
    slot: &mut ThrmlSlot,
) -> SideMeasure {
    let t0 = std::time::Instant::now();
    slot.arm(side);
    let (reps, ess_sum) = repeat_until_window(seed, |seed_i| {
        plan_ess(side, w, seed_i, slot.session.as_mut())
    });
    let seconds = t0.elapsed().as_secs_f64();
    std::thread::sleep(std::time::Duration::from_millis(250));
    integrate_logged(series, t0, reps, ess_sum, seconds)
}

fn integrate_logged(
    series: &energy::HwmonSeries,
    t0: std::time::Instant,
    reps: u32,
    ess_sum: f64,
    seconds: f64,
) -> SideMeasure {
    let stamped = series.since_stamped(t0);
    let samples = u32::try_from(stamped.len()).unwrap_or(u32::MAX);
    if series.counter {
        let inside: Vec<f64> = stamped
            .iter()
            .filter(|(t, _)| *t >= t0)
            .map(|(_, j)| *j)
            .collect();
        let gross = match (inside.first(), inside.last()) {
            (Some(a), Some(b)) if b >= a => b - a,
            _ => f64::NAN,
        };
        let cadence = if inside.len() > 1 {
            seconds / (inside.len() - 1) as f64
        } else {
            f64::NAN
        };
        return finish_window(
            ess_sum,
            gross,
            seconds,
            seconds,
            reps,
            samples.saturating_sub(1),
            series.instrument,
            series.scope.as_str(),
            cadence,
            "counter",
            "counter",
            0.0,
            samples,
            Some(energy::counter_quantization_j(1e-6)),
            energy::counter_cadence_edge_j(gross, seconds, cadence),
        );
    }
    let points = series_secs(t0, &stamped);
    let Ok(integ) = ferrotherm_meter::trapezoid(&points, 0.0, seconds) else {
        return empty_side(reps, series.instrument, &series.scope);
    };
    let n = u32::try_from(integ.samples).unwrap_or(u32::MAX);
    finish_window(
        ess_sum,
        integ.joules,
        integ.covered_s,
        seconds,
        reps,
        n,
        series.instrument,
        series.scope.as_str(),
        integ.cadence_s,
        integ.left.label(),
        integ.right.label(),
        integ.uncovered_left_s + integ.uncovered_right_s,
        n,
        None,
        integ.cadence_edge_j(),
    )
}

fn measure_wall(
    probe: &mut Probe,
    side: &str,
    w: &Workload,
    seed: u64,
    slot: &mut ThrmlSlot,
) -> SideMeasure {
    let wall = probe.wall.as_mut().expect("wall was checked");
    let backend = wall.backend();
    let (instrument, scope) = if backend == "macmon" {
        ("macmon", "soc-telemetry")
    } else {
        ("ina3221", "board")
    };
    let mut reps = 0u32;
    let mut ess_sum = 0.0;
    let integrated = wall.integrate(
        Baseline {
            watts: 0.0,
            sigma: 0.0,
            samples: 0,
            load1: None,
            cadence_s: f64::NAN,
        },
        || {
            slot.arm(side);
            let (r, e) = repeat_until_window(seed, |seed_i| {
                plan_ess(side, w, seed_i, slot.session.as_mut())
            });
            reps = r;
            ess_sum = e;
        },
    );
    // The zero baseline above is only so integrate returns a Run. Gross is the trapezoid
    // over the covered span. Incremental is applied by the caller against the real idle.
    // Joules and ESS below are per repetition.
    match integrated {
        Ok((run, ())) => finish_window(
            ess_sum,
            run.joules_total,
            run.covered_s,
            run.seconds,
            reps,
            u32::try_from(run.samples).unwrap_or(u32::MAX),
            instrument,
            scope,
            run.cadence_s,
            run.left_edge.label(),
            run.right_edge.label(),
            run.uncovered_s,
            u32::try_from(run.samples).unwrap_or(u32::MAX),
            None,
            run.cadence_edge_j,
        ),
        Err(_) => empty_side(reps, instrument, scope),
    }
}

enum CounterKind {
    Rapl,
    Perf,
    Nvml,
}

fn measure_tegra(
    probe: &mut Probe,
    side: &str,
    w: &Workload,
    seed: u64,
    slot: &mut ThrmlSlot,
) -> SideMeasure {
    let t0 = std::time::Instant::now();
    slot.arm(side);
    let (reps, ess_sum) = repeat_until_window(seed, |seed_i| {
        plan_ess(side, w, seed_i, slot.session.as_mut())
    });
    let seconds = t0.elapsed().as_secs_f64();
    std::thread::sleep(std::time::Duration::from_millis(250));
    let scope = if probe.tegra.as_ref().is_some_and(|t| t.label == "VIN") {
        "vin"
    } else {
        "board"
    };
    let stamped = probe
        .tegra
        .as_ref()
        .map(|t| t.since_stamped(t0))
        .unwrap_or_default();
    let series = series_secs(t0, &stamped);
    let Ok(integ) = ferrotherm_meter::trapezoid(&series, 0.0, seconds) else {
        return empty_side(reps, "tegrastats", scope);
    };
    let samples = u32::try_from(integ.samples).unwrap_or(u32::MAX);
    finish_window(
        ess_sum,
        integ.joules,
        integ.covered_s,
        seconds,
        reps,
        samples,
        "tegrastats",
        scope,
        integ.cadence_s,
        integ.left.label(),
        integ.right.label(),
        integ.uncovered_left_s + integ.uncovered_right_s,
        samples,
        None,
        integ.cadence_edge_j(),
    )
}

fn measure_counter(
    probe: &mut Probe,
    side: &str,
    w: &Workload,
    seed: u64,
    slot: &mut ThrmlSlot,
    kind: CounterKind,
) -> Vec<SideMeasure> {
    let core_path = if matches!(kind, CounterKind::Rapl) {
        probe
            .rapl_core
            .as_ref()
            .map(|c| c.counter_path().to_path_buf())
    } else {
        None
    };
    let core_max = probe
        .rapl_core
        .as_ref()
        .map_or(0.0, energy::RaplCounter::range_j);
    let core_log = std::sync::Arc::new(std::sync::Mutex::new(Vec::<f64>::new()));
    let opened = match kind {
        CounterKind::Rapl => {
            let rapl = probe.rapl.as_ref().expect("rapl was checked");
            let scope = if rapl.kind == "psys" {
                "psys"
            } else {
                "package"
            };
            let max_j = rapl.range_j();
            let path = rapl.counter_path().to_path_buf();
            let sink = std::sync::Arc::clone(&core_log);
            let core_for_read = core_path.clone();
            OpenedCounter {
                instrument: "rapl",
                scope,
                gross: Gross::Joules { max_j: Some(max_j) },
                read: Box::new(move || {
                    if let Some(core) = &core_for_read
                        && let Some(j) = read_energy_uj(core)
                    {
                        sink.lock().unwrap().push(j);
                    }
                    read_energy_uj(&path)
                }),
            }
        }
        CounterKind::Perf => {
            let (fd, scale) = probe
                .perf
                .as_ref()
                .expect("perf was checked")
                .fd_and_scale();
            OpenedCounter {
                instrument: "perf",
                scope: "package",
                gross: Gross::Joules { max_j: None },
                read: Box::new(move || energy::perf_sample_j(fd, scale)),
            }
        }
        CounterKind::Nvml => {
            let sampler = probe.nvml.as_ref().expect("nvml was checked").sampler();
            OpenedCounter {
                instrument: "nvml",
                scope: "gpu",
                gross: Gross::Millijoules,
                read: Box::new(move || sampler.millijoules().map(|mj| mj as f64)),
            }
        }
    };
    let sampled = energy::sample_during(opened.read, || {
        slot.arm(side);
        repeat_until_window(seed, |seed_i| {
            plan_ess(side, w, seed_i, slot.session.as_mut())
        })
    });
    let readings = sampled.readings;
    let window_s = sampled.seconds;
    let (reps, ess_sum) = sampled.value;
    let gross_window = match opened.gross {
        Gross::Joules { max_j } => energy::window_gross(&readings, max_j),
        Gross::Millijoules => energy::mj_window_j(&readings),
    };
    let samples = u32::try_from(readings.len()).unwrap_or(u32::MAX);
    let quantum = energy::counter_quantum_j(opened.instrument).map(energy::counter_quantization_j);
    let mut out = vec![finish_window(
        ess_sum,
        gross_window,
        window_s,
        window_s,
        reps,
        energy::count_counter_updates(&readings),
        opened.instrument,
        opened.scope,
        sampled.cadence_s,
        "counter",
        "counter",
        0.0,
        samples,
        quantum,
        energy::counter_cadence_edge_j(gross_window, window_s, sampled.cadence_s),
    )];
    if core_path.is_some() {
        let samples = core_log.lock().unwrap().clone();
        let core_gross = energy::window_gross(&samples, Some(core_max));
        out.push(finish_window(
            ess_sum,
            core_gross,
            window_s,
            window_s,
            reps,
            energy::count_counter_updates(&samples),
            "rapl",
            "core",
            sampled.cadence_s,
            "counter",
            "counter",
            0.0,
            u32::try_from(samples.len()).unwrap_or(u32::MAX),
            energy::counter_quantum_j("rapl").map(energy::counter_quantization_j),
            energy::counter_cadence_edge_j(core_gross, window_s, sampled.cadence_s),
        ));
    }
    out
}

fn read_energy_uj(dir: &std::path::Path) -> Option<f64> {
    let text = std::fs::read_to_string(dir.join("energy_uj")).ok()?;
    let uj: u64 = text.trim().parse().ok()?;
    Some(uj as f64 / 1e6)
}

type CounterRead = Box<dyn FnMut() -> Option<f64> + Send>;

struct OpenedCounter {
    instrument: &'static str,
    scope: &'static str,
    gross: Gross,
    read: CounterRead,
}

enum Gross {
    Joules { max_j: Option<f64> },
    Millijoules,
}

fn finish_window(
    ess_sum: f64,
    gross_window: f64,
    energy_seconds: f64,
    window_seconds: f64,
    repetitions: u32,
    counter_updates: u32,
    instrument: &'static str,
    scope: &str,
    cadence_s: f64,
    left_edge: &'static str,
    right_edge: &'static str,
    uncovered_s: f64,
    samples: u32,
    quantization_j: Option<f64>,
    cadence_edge_j: f64,
) -> SideMeasure {
    let (gross, seconds) = energy::per_repetition(gross_window, energy_seconds, repetitions);
    let ess = energy::per_repetition(ess_sum, energy_seconds, repetitions).0;
    SideMeasure {
        ess,
        gross,
        seconds,
        window_seconds,
        repetitions,
        counter_updates,
        instrument,
        scope: scope.to_string(),
        cadence_s,
        left_edge,
        right_edge,
        uncovered_s,
        samples,
        quantization_j,
        cadence_edge_j,
        window_load1: String::new(),
        window_signal: String::new(),
        ps_attribution: String::new(),
        trial_refusal: String::new(),
    }
}

fn empty_side(repetitions: u32, instrument: &'static str, scope: &str) -> SideMeasure {
    SideMeasure {
        ess: f64::NAN,
        gross: f64::NAN,
        seconds: 0.0,
        window_seconds: 0.0,
        repetitions,
        counter_updates: 0,
        instrument,
        scope: scope.to_string(),
        cadence_s: f64::NAN,
        left_edge: "uncovered",
        right_edge: "uncovered",
        uncovered_s: 0.0,
        samples: 0,
        quantization_j: None,
        cadence_edge_j: f64::NAN,
        window_load1: String::new(),
        window_signal: String::new(),
        ps_attribution: String::new(),
        trial_refusal: String::new(),
    }
}

fn series_secs(
    origin: std::time::Instant,
    samples: &[(std::time::Instant, f64)],
) -> Vec<(f64, f64)> {
    samples
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

fn repeat_until_window(seed: u64, mut once: impl FnMut(u64) -> f64) -> (u32, f64) {
    let started = Instant::now();
    let mut reps = 0u32;
    let mut ess = 0.0;
    while reps == 0 || started.elapsed().as_secs_f64() < MIN_ENERGY_WINDOW_S {
        ess += once(seed.wrapping_add(u64::from(reps)));
        reps = reps.saturating_add(1);
    }
    (reps, ess)
}

fn plan_ess(side: &str, w: &Workload, seed: u64, session: Option<&mut ThrmlServe>) -> f64 {
    one_plan(side, w, seed, session)
}

struct ThrmlSlot {
    session: Option<ThrmlServe>,
    side: &'static str,
    baseline: &'static str,
    python: PathBuf,
    runner: PathBuf,
    spec: PathBuf,
    device: String,
    enabled: bool,
    /// Ex-jit nanoseconds per sweep from DONE lines in this side's window.
    ex_jit_ns: Vec<f64>,
    oom: bool,
}

impl ThrmlSlot {
    /// Start THRML on the first B-arm call. The caller invokes this inside the timed window.
    fn arm(&mut self, side: &str) {
        if side == self.side && self.enabled && self.session.is_none() {
            match ThrmlServe::start(
                &self.python,
                &self.runner,
                &self.spec,
                &self.device,
                self.baseline,
            ) {
                Ok(child) => self.session = Some(child),
                Err(e) => {
                    if e.contains("THRML_LIMIT") {
                        self.oom = true;
                    }
                    eprintln!("THRML start inside the trial window failed: {e}");
                }
            }
        }
    }

    /// Drop the running process and point the next start at `spec`. The process is not
    /// started here; `arm` starts it inside the trial window.
    fn restart(&mut self, spec: &Path) {
        self.session = None;
        self.spec = spec.to_path_buf();
        self.oom = false;
        self.ex_jit_ns.clear();
    }
}

fn one_plan(side: &str, w: &Workload, seed: u64, session: Option<&mut ThrmlServe>) -> f64 {
    if w.open_grid().is_some() {
        // Cost only. ESS is measured after the window, on the same seeds.
        if side == "ferrotherm" {
            let (levers, threads, gpu_on) = GRID_CHOICE.with(|c| c.get());
            if gpu_on {
                let _ = gpu_cost(w, seed);
            } else {
                let _ = ft::cost_sweeps(w, seed, Some(ft::GridChoice { levers, threads }));
            }
        } else if let Some(s) = session {
            match s.cost(w, seed) {
                Ok(Some((seconds, sweeps))) if sweeps > 0 && seconds.is_finite() => {
                    EXJIT_NS.with(|v| v.borrow_mut().push(seconds * 1.0e9 / sweeps as f64));
                }
                Ok(_) => {}
                Err(e) if e.contains("THRML_LIMIT") => EXJIT_OOM.with(|c| c.set(true)),
                Err(e) => eprintln!("THRML cost: {e}"),
            }
        }
        return f64::NAN;
    }
    if side == "ferrotherm" {
        ft::run(w, seed).map_or(f64::NAN, |c| c.score.ess_energy)
    } else if let Some(s) = session {
        s.run(w, seed).map_or(f64::NAN, |c| c.score.ess_energy)
    } else {
        f64::NAN
    }
}

thread_local! {
    static GRID_CHOICE: std::cell::Cell<(ferrotherm::grid::Levers, usize, bool)> =
        std::cell::Cell::new((ferrotherm::grid::Levers::SCALAR, 1, false));
    static EXJIT_NS: std::cell::RefCell<Vec<f64>> = const { std::cell::RefCell::new(Vec::new()) };
    static EXJIT_OOM: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn remix_ess(side: &str, w: &Workload, seed: u64, reps: u32, slot: &mut ThrmlSlot) -> f64 {
    if reps == 0 {
        return f64::NAN;
    }
    let mut sum = 0.0;
    let mut n = 0u32;
    for r in 0..reps {
        let s = seed.wrapping_add(u64::from(r));
        let ess = if side == "ferrotherm" {
            let (levers, threads, gpu_on) = GRID_CHOICE.with(|c| c.get());
            if gpu_on {
                run_gpu_chain(w, s)
                    .map(|c| c.score.ess_energy)
                    .unwrap_or(f64::NAN)
            } else {
                ft::run_with(w, s, Some(ft::GridChoice { levers, threads }))
                    .map(|c| c.score.ess_energy)
                    .unwrap_or(f64::NAN)
            }
        } else {
            slot.arm(side);
            match slot.session.as_mut() {
                Some(session) => session.ess(w, s).unwrap_or(f64::NAN),
                None => f64::NAN,
            }
        };
        if ess.is_finite() {
            sum += ess;
            n += 1;
        }
    }
    if n == 0 { f64::NAN } else { sum / f64::from(n) }
}

fn gpu_cost(w: &Workload, seed: u64) -> Result<(), String> {
    #[cfg(feature = "gpu")]
    {
        let (side, j) = w
            .open_grid()
            .ok_or_else(|| format!("{} is not an open grid", w.id))?;
        let sweeps = u32::try_from(w.warmup + w.draws * w.thin.max(1)).unwrap_or(u32::MAX);
        let mut state = vec![-1i8; side * side];
        with_gpu(|gpu| gpu.sweep_packed(side, j, w.beta, &mut state, sweeps, seed, 0))?;
        Ok(())
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = (w, seed);
        Err("gpu feature is off".into())
    }
}

fn profile_json(
    work: &[Workload],
    choice: Option<ft::GridChoice>,
    sweeps: usize,
    gpu: bool,
) -> String {
    let mut s = String::from("[\n");
    for (i, w) in work.iter().enumerate() {
        let Some((side, j)) = w.open_grid() else {
            continue;
        };
        if i > 0 && !s.ends_with("[\n") {
            s.push_str(",\n");
        }
        if gpu {
            s.push_str(&format!(
                "    {{\"workload\": {}, \"side\": {side}, \"fabric\": \"ferrotherm-gpu\", \
                 \"note\": \"The chain is one submission. sweeps_per_submit and wall_ns_per_sweep are on the accuracy trial. setup_ns is pipeline creation and is inside that wall time. Not joules.\"}}",
                json_str(w.id)
            ));
            continue;
        }
        let choice = choice.unwrap_or(ft::GridChoice {
            levers: ferrotherm::grid::Levers::SCALAR,
            threads: 1,
        });
        let mut g = match ferrotherm::grid::OpenIsing::try_new(
            side,
            j,
            w.beta,
            w.seed0,
            choice.levers,
            choice.threads,
        ) {
            Ok(g) => g,
            Err(e) => {
                s.push_str(&format!(
                    "    {{\"workload\": {}, \"error\": {}}}",
                    json_str(w.id),
                    json_str(&e.message)
                ));
                continue;
            }
        };
        g.set_profile(true);
        let prof = g.sweeps(sweeps);
        let wall_per = if prof.sweeps > 0 {
            prof.wall_ns as f64 / prof.sweeps as f64
        } else {
            f64::NAN
        };
        s.push_str(&format!(
            "    {{\"workload\": {}, \"side\": {side}, \"fabric\": \"ferrotherm-cpu\", \
             \"levers\": {{\"parallel\": {}, \"simd\": {}, \"tiled\": {}, \"sparse_rng\": {}, \"incremental\": {}}}, \
             \"threads_used\": {}, \"sweeps\": {}, \"wall_ns\": {}, \"wall_ns_per_sweep\": {:.4}, \
             \"memory_ns\": {}, \"field_ns\": {}, \"rng_ns\": {}, \"rng_words\": {}, \
             \"note\": \"memory_ns, field_ns and rng_ns are CPU time summed across threads. wall_ns is elapsed. Not joules.\"}}",
            json_str(w.id),
            if choice.levers.parallel { "true" } else { "false" },
            if choice.levers.simd { "true" } else { "false" },
            if choice.levers.tiled { "true" } else { "false" },
            if choice.levers.sparse_rng { "true" } else { "false" },
            if choice.levers.incremental { "true" } else { "false" },
            prof.threads_used,
            prof.sweeps,
            prof.wall_ns,
            wall_per,
            prof.memory_ns,
            prof.field_ns,
            prof.rng_ns,
            prof.rng_words,
        ));
    }
    s.push_str("\n  ]");
    s
}

fn fabric_name(side: &str, pairing: &str, device: &str, gpu: bool) -> &'static str {
    let cuda = device == "cuda";
    if side == "ferrotherm" {
        if gpu {
            "ferrotherm-gpu"
        } else {
            "ferrotherm-cpu"
        }
    } else if pairing == "B" {
        if cuda {
            "thrml-main-2d96f4b4-jax-cuda"
        } else {
            "thrml-main-2d96f4b4-jax-cpu"
        }
    } else if cuda {
        "thrml-0.1.4-jax-cuda"
    } else {
        "thrml-0.1.4-jax-cpu"
    }
}

fn trial_window_rule_json(rule: &energy::TrialWindowRule) -> String {
    format!(
        "{{\"enforce\": {enforce}, \"other_core_mean_max\": {mean}, \"other_core_sample_max\": {sample}, \
          \"foreign_cpu_mean_max\": {foreign_mean}, \"foreign_cpu_sample_max\": {foreign_sample}, \
          \"why_load1_is_not_a_trial_criterion\": {why}}}",
        enforce = if rule.enforce { "true" } else { "false" },
        mean = rule.other_core_mean_max,
        sample = rule.other_core_sample_max,
        foreign_mean = rule.foreign_cpu_mean_max,
        foreign_sample = rule.foreign_cpu_sample_max,
        why = json_str(
            "load1 counts runnable threads, so the multithreaded workload under test trips it. \
             Idle brackets use load1. A trial window uses the busy fraction of cores outside \
             --cpus on Linux, and the summed CPU of processes that are not this harness on macOS. \
             Those fractions attribute and refuse. They are not subtracted from joules.",
        ),
    )
}

fn energy_json(
    probe: &Probe,
    poll: &energy::QuietPoll,
    grade: &str,
    outcome: &str,
    idle: Option<(f64, f64)>,
    rows: &[EnergyTrial],
    note: &str,
    rule: &energy::TrialWindowRule,
    baseline_settle: &[f64],
) -> String {
    let summaries = energy::summarise_workloads(rows);
    format!(
        "{{\n    \"evidence_grade\": {},\n    \"outcome\": {},\n    \"quiet_threshold\": {},\n    \
         \"load_samples\": {},\n    \"rank_policy\": {},\n    \"order\": {},\n    \
         \"window_s\": {},\n    \"min_counter_updates\": {},\n    \
         \"backends\": {{\"macmon_or_ina\": {}, \"ina238\": {}, \"tegrastats\": {}, \"rapl\": {}, \"rapl_core\": {}, \"perf\": {}, \"nvml\": {}}},\n    \
         \"hostname\": {},\n    \"macos_policy\": {},\n    \"accepted_load\": {},\n    \
         \"idle_watts\": {},\n    \"idle_sigma\": {},\n    \
         \"trial_window_rule\": {},\n    \"baseline_settle_load1\": {},\n    \
         \"summaries\": [\n{}    ],\n    \"trials\": [\n{}    ],\n    \
         \"projected_prices_applied\": false,\n    \"note\": {}\n  }}",
        json_str(grade),
        json_str(outcome),
        poll.threshold,
        loads_json(&poll.samples),
        json_str(
            "primary = RAPL package on framework-128gb-0000, read without root; macmon sys_power = SoC telemetry, secondary; package and core are not combined",
        ),
        json_str(
            "paired A/B then B/A; idle bracket before and after each pair; joules per repetition of a >=2 s window"
        ),
        MIN_ENERGY_WINDOW_S,
        energy::MIN_COUNTER_UPDATES,
        json_str(&probe.wall_note),
        json_str(&probe.ina238_note),
        json_str(&probe.tegra_note),
        json_str(&probe.rapl_note),
        json_str(&probe.rapl_core_note),
        json_str(&probe.perf_note),
        json_str(&probe.nvml_note),
        json_str(&probe.hostname),
        json_str(probe.macos_policy),
        fmt_f64(poll.accepted),
        fmt_f64(idle.map(|p| p.0)),
        fmt_f64(idle.map(|p| p.1)),
        trial_window_rule_json(rule),
        loads_json(baseline_settle),
        summaries
            .iter()
            .map(grade_summary_json)
            .collect::<Vec<_>>()
            .join(",\n"),
        rows.iter().map(trial_line).collect::<Vec<_>>().join(",\n"),
        json_str(note),
    )
}

fn loads_json(loads: &[f64]) -> String {
    let parts: Vec<String> = loads
        .iter()
        .map(|l| {
            if l.is_finite() {
                format!("{l:.4}")
            } else {
                "null".into()
            }
        })
        .collect();
    format!("[{}]", parts.join(", "))
}

fn grade_summary_json(s: &GradeSummary) -> String {
    format!(
        "      {{\"workload\": {}, \"side\": {}, \"pairing\": {}, \"fabric\": {}, \"instrument\": {}, \"scope\": {}, \"grade\": {}, \
         \"n\": {}, \"gross_j\": {}, \"incremental_j\": {}, \"joules_per_ess\": {}, \
         \"uncertainty_j_per_ess\": {}, \"scope_limit\": {}}}",
        json_str(s.workload),
        json_str(s.side),
        json_str(s.pairing),
        json_str(s.fabric),
        json_str(&s.instrument),
        json_str(&s.scope),
        json_str(s.grade),
        s.n,
        ci_opt(s.gross),
        ci_opt(s.incremental),
        ci_opt(s.joules_per_ess),
        uncertainty_json(&s.uncertainty),
        json_str(energy::scope_limit(&s.scope)),
    )
}

fn uncertainty_json(u: &energy::UncertaintyBudget) -> String {
    format!(
        "{{\"student_t_half\": {}, \"quantization\": {}, \"cadence_edge\": {}, \"bracket\": {}, \"combined\": {}}}",
        fmt_f64(Some(u.student_t_half)),
        fmt_f64(u.quantization),
        fmt_f64(u.cadence_edge),
        fmt_f64(Some(u.bracket)),
        fmt_f64(Some(u.combined)),
    )
}

fn ci_opt(v: Option<(f64, f64, f64, usize)>) -> String {
    match v {
        Some((mean, lo, hi, n)) => {
            format!("{{\"mean\": {mean:.8e}, \"lo95\": {lo:.8e}, \"hi95\": {hi:.8e}, \"n\": {n}}}")
        }
        None => "null".into(),
    }
}

fn trial_line(t: &EnergyTrial) -> String {
    format!(
        "      {{\"workload\": {}, \"side\": {}, \"trial\": {}, \"repetitions\": {}, \
         \"counter_updates\": {}, \"samples\": {}, \"cadence_s\": {}, \"left_edge\": {}, \
         \"right_edge\": {}, \"uncovered_s\": {}, \"window_seconds\": {:.6}, \"grade\": {}, \"rank\": {}, \
         \"load1\": {}, \"other_process_cpu_s\": {}, \"other_core_busy\": {}, \
         \"idle_before_w\": {}, \"idle_after_w\": {}, \"baseline_drift_w\": {}, \
         \"paired_delta_j\": {}, \"paired_attributable\": {}, \"ess\": {}, \
         \"joules_per_ess\": {}, \"sweeps_per_rep\": {}, \"ess_per_sweep\": {}, \
         \"seconds_per_sweep\": {}, \"wall_ns_per_sweep\": {}, \"joules_per_sweep\": {}, \
         \"pairing\": {}, \"fabric\": {}, \
         \"window_load1\": {}, \"window_signal\": {}, \
         \"ps_attribution\": {}, \"trial_refusal\": {}, \"settle_load1\": {}, \
         \"joules\": {}}}",
        json_str(t.workload),
        json_str(t.side),
        t.trial,
        t.repetitions,
        t.counter_updates,
        t.samples,
        fmt_f64(Some(t.cadence_s)),
        json_str(&t.left_edge),
        json_str(&t.right_edge),
        fmt_f64(Some(t.uncovered_s)),
        t.window_seconds,
        json_str(t.grade),
        json_str(t.rank),
        fmt_f64(t.load1),
        fmt_f64(t.other_cpu_s),
        json_str(&t.other_core_busy),
        fmt_f64(t.idle_before_w),
        fmt_f64(t.idle_after_w),
        fmt_f64(t.baseline_drift_w),
        fmt_f64(t.paired_delta_j),
        if t.paired_attributable {
            "true"
        } else {
            "false"
        },
        fmt_f64(Some(t.ess)),
        fmt_f64(t.joules_per_ess),
        t.sweeps_per_rep,
        fmt_f64(Some(t.ess_per_sweep)),
        fmt_f64(Some(t.seconds_per_sweep)),
        fmt_f64(Some(t.wall_ns_per_sweep)),
        fmt_f64(t.joules_per_sweep),
        json_str(t.pairing),
        json_str(t.fabric),
        json_str(&t.window_load1),
        json_str(&t.window_signal),
        json_str(&t.ps_attribution),
        json_str(&t.trial_refusal),
        json_str(&t.settle_load1),
        joules_json(&t.joules),
    )
}

struct ThrmlServe {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    out: BufReader<std::process::ChildStdout>,
}

impl ThrmlServe {
    fn start(
        python: &Path,
        runner: &Path,
        spec: &Path,
        device: &str,
        baseline: &str,
    ) -> Result<ThrmlServe, String> {
        let mut child = Command::new(python)
            .arg(runner)
            .arg("--serve")
            .arg("--device")
            .arg(device)
            .arg("--baseline")
            .arg(baseline)
            .arg("--spec")
            .arg(spec)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .map_err(|e| format!("THRML serve: {e}"))?;
        let stdin = child.stdin.take().ok_or("THRML serve has no stdin")?;
        let stdout = child.stdout.take().ok_or("THRML serve has no stdout")?;
        let mut out = BufReader::new(stdout);
        let mut line = String::new();
        let ready_read = out.read_line(&mut line);
        if ready_read.map(|n| n == 0).unwrap_or(true) || !line.starts_with("READY") {
            let status = child.wait().ok();
            let killed = status.as_ref().and_then(limit_from_status).is_some();
            if killed || thrml_oom_text(&line) {
                return Err(thrml_limit_reason("oom"));
            }
            return Err(format!("THRML serve said {line:?}"));
        }
        Ok(ThrmlServe { child, stdin, out })
    }

    fn run(&mut self, w: &Workload, seed: u64) -> Result<Chain, String> {
        writeln!(self.stdin, "RUN {} {seed}", w.id)
            .map_err(|e| format!("THRML serve write: {e}"))?;
        self.stdin.flush().ok();
        self.read_chain(w)
    }

    fn read_chain(&mut self, w: &Workload) -> Result<Chain, String> {
        let mut buf = String::new();
        loop {
            let mut line = String::new();
            let n = self
                .out
                .read_line(&mut line)
                .map_err(|e| format!("THRML serve read: {e}"))?;
            if n == 0 {
                let status = self.child.try_wait().ok().flatten();
                if status.as_ref().and_then(limit_from_status).is_some() {
                    return Err(thrml_limit_reason("oom"));
                }
                return Err("THRML serve closed".into());
            }
            if line.starts_with("LIMIT") || thrml_oom_text(&line) {
                return Err(thrml_limit_reason("oom"));
            }
            if line.starts_with("ERR") {
                return Err(line.trim().to_string());
            }
            buf.push_str(&line);
            if line.trim() == "END" {
                break;
            }
        }
        let (_, mut map, _) = parse_thrml(&buf)?;
        let raw = map
            .remove(w.id)
            .and_then(|mut v| v.pop())
            .ok_or("THRML serve returned no chain")?;
        score_external(w, &raw)
    }

    fn ess(&mut self, w: &Workload, seed: u64) -> Result<f64, String> {
        writeln!(self.stdin, "ESS {} {seed}", w.id).map_err(|e| format!("THRML ess write: {e}"))?;
        self.stdin.flush().ok();
        let chain = self.read_chain(w)?;
        Ok(chain.score.ess_energy)
    }

    /// Seconds and sweeps from the DONE line. Seconds exclude JIT: the serve process
    /// compiled before READY, and this timer starts at the sweep.
    fn cost(&mut self, w: &Workload, seed: u64) -> Result<Option<(f64, u64)>, String> {
        writeln!(self.stdin, "COST {} {seed}", w.id)
            .map_err(|e| format!("THRML cost write: {e}"))?;
        self.stdin.flush().ok();
        let mut line = String::new();
        let n = self
            .out
            .read_line(&mut line)
            .map_err(|e| format!("THRML cost read: {e}"))?;
        if n == 0 {
            let status = self.child.wait().ok();
            if status.as_ref().and_then(limit_from_status).is_some() {
                return Err(thrml_limit_reason("oom"));
            }
            return Err("THRML serve closed".into());
        }
        if line.starts_with("LIMIT") || thrml_oom_text(&line) {
            return Err(thrml_limit_reason("oom"));
        }
        if !line.starts_with("DONE") {
            return Err(format!("THRML cost said {}", line.trim()));
        }
        let mut seconds = None;
        let mut sweeps = None;
        for part in line.split_whitespace() {
            if let Some(v) = part.strip_prefix("seconds=") {
                seconds = v.parse().ok();
            } else if let Some(v) = part.strip_prefix("sweeps=") {
                sweeps = v.parse().ok();
            }
        }
        match (seconds, sweeps) {
            (Some(seconds), Some(sweeps)) => Ok(Some((seconds, sweeps))),
            _ => Err(format!("THRML cost line had no seconds and sweeps: {}", line.trim())),
        }
    }
}

impl Drop for ThrmlServe {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "QUIT");
        let _ = self.child.wait();
    }
}

fn joules_json(j: &Joules) -> String {
    format!(
        "{{\"gross_j\": {}, \"incremental_j\": {}, \"idle_watts\": {}, \"idle_sigma\": {}, \
         \"seconds\": {:.6}, \"attributable\": {}, \"evidence\": {}, \"instrument\": {}, \
         \"scope\": {}, \"note\": {}, \"projected_prices_applied\": {}}}",
        fmt_f64(j.gross),
        fmt_f64(j.incremental),
        fmt_f64(j.idle_watts),
        fmt_f64(j.idle_sigma),
        j.seconds,
        if j.attributable { "true" } else { "false" },
        json_str(j.evidence),
        json_str(&j.instrument),
        json_str(&j.scope),
        json_str(&j.note),
        if j.projected_prices_applied {
            "true"
        } else {
            "false"
        },
    )
}

fn phases_json(preflight: &str, energy_grade: &str, accuracy: &str) -> String {
    format!(
        "{{\"preflight\": {{\"grade\": {}}}, \"energy\": {{\"grade\": {}}}, \"accuracy\": {{\"grade\": {}}}}}",
        json_str(preflight),
        json_str(energy_grade),
        json_str(accuracy),
    )
}

fn host_lock_json(
    path: &Path,
    acquired: Option<&str>,
    released: Option<&str>,
    wait_s: f64,
    holder: Option<u32>,
    held: bool,
    reason: Option<&str>,
) -> String {
    let acquired_json = match acquired {
        Some(text) => json_str(text),
        None => "null".into(),
    };
    let released_json = match released {
        Some(text) => json_str(text),
        None => "null".into(),
    };
    let holder_json = match holder {
        Some(pid) => pid.to_string(),
        None => "null".into(),
    };
    let reason_json = match reason {
        Some(text) => json_str(text),
        None => "null".into(),
    };
    format!(
        "{{\"path\": {}, \"acquired_utc\": {acquired_json}, \"released_utc\": {released_json}, \
          \"wait_s\": {wait_s:.3}, \"holder_pid\": {holder_json}, \"held\": {}, \"reason\": {reason_json}}}",
        json_str(&path.display().to_string()),
        if held { "true" } else { "false" },
    )
}

fn release_lock_json(lock: energy::HostLock) -> String {
    let path = lock.path.clone();
    let acquired = lock.acquired_utc();
    let wait_s = lock.wait_s;
    let holder = lock.holder_pid;
    let released = lock.release();
    host_lock_json(
        &path,
        Some(&acquired),
        Some(&released),
        wait_s,
        Some(holder),
        true,
        None,
    )
}

fn write_lock_refusal(out: &Path, path: &Path, wait_s: f64) {
    let reason = energy::HOST_LOCK_REFUSAL;
    let energy = format!(
        "{{\"evidence_grade\": \"load_gated\", \"outcome\": \"refused\", \"reason\": {}, \"note\": {}}}",
        json_str(reason),
        json_str(reason),
    );
    let report = format!(
        "{{\n  \"schema\": \"ferrotherm-thrml-bench/1\",\n  \"evidence_grade\": \"load_gated\",\n  \
         \"phases\": {phases},\n  \"host_lock\": {lock},\n  \"energy\": {energy},\n  \"workloads\": []\n}}\n",
        phases = phases_json("not_run", "load_gated", "not_run"),
        lock = host_lock_json(path, None, None, wait_s, None, false, Some(reason)),
    );
    fs::write(out.join("report.json"), report).expect("write report");
    eprintln!("{reason}");
}

fn quiet_gate_json(gate: &energy::QuietGate) -> String {
    let busy: Vec<String> = gate
        .other_core_busy
        .iter()
        .map(|(cpu, busy)| format!("{{\"cpu\": {cpu}, \"busy\": {busy:.6}}}"))
        .collect();
    format!(
        "{{\"seconds\": {:.3}, \"threshold\": {}, \"held\": {}, \"evidence_grade\": {}, \
         \"max_load1\": {}, \"busy_readable\": {}, \"load1\": {}, \"other_core_busy\": [{}]}}",
        gate.seconds,
        gate.threshold,
        if gate.held { "true" } else { "false" },
        json_str(if gate.held {
            energy::GRADE_QUIET
        } else {
            "load_gated"
        }),
        fmt_f64(gate.max_load1),
        if gate.busy_readable { "true" } else { "false" },
        loads_json(&gate.samples),
        busy.join(", "),
    )
}

fn provenance_json(probe: &Probe, grade: &str) -> String {
    let rev = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok();
    let rev = rev.filter(|o| o.status.success()).map_or_else(
        || "unknown".into(),
        |o| String::from_utf8_lossy(&o.stdout).trim().to_string(),
    );
    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok();
    let dirty = dirty
        .filter(|o| o.status.success())
        .is_some_and(|o| !o.stdout.is_empty());
    let cpu = fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|v| v.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".into());
    format!(
        "{{\n    \"evidence_grade\": {},\n    \"git_rev\": {},\n    \"git_dirty\": {},\n    \"host_cpu\": {},\n    \
         \"os\": {},\n    \"rustc\": {},\n    \"thrml_pin\": {},\n    \"thrml_wheel_sha256\": {},\n    \
         \"load1_at_start\": {},\n    \"what_this_machine_measured\": {},\n    \
         \"what_still_needs_fleet_hardware\": {}\n  }}",
        json_str(grade),
        json_str(&rev),
        if dirty { "true" } else { "false" },
        json_str(&cpu),
        json_str(std::env::consts::OS),
        json_str(&rustc_version()),
        json_str(&format!("thrml==0.1.4 and {}", spec::THRML_B_COMMIT)),
        json_str("6e2f38cecb562589d230ca063b5fcb5d2a6533201e37bb70c1f2dac4a63a0858"),
        match probe.load1 {
            Some(v) => format!("{v:.4}"),
            None => "null".into(),
        },
        json_str(&format!(
            "wall: {}. nvml: {}.",
            probe.wall_note, probe.nvml_note
        )),
        json_str(
            "A quiet machine with a live RAPL domain (or a whole-board rail) for CPU joules; \
             an NVIDIA GPU with libnvidia-ml.so.1 for GPU joules; an Apple silicon host with \
             macmon on PATH for the SoC counter. None of those counters were a substitute for \
             the others, and none of them is a Z1 device. BMI fleet hardware is what would \
             meter a sustained run and, separately, any future silicon.",
        ),
    )
}

fn rustc_version() -> String {
    Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map_or_else(
            || "unknown".into(),
            |o| String::from_utf8_lossy(&o.stdout).trim().to_string(),
        )
}

fn fmt_finite(x: f64) -> String {
    if x.is_finite() {
        format!("{x:.8e}")
    } else {
        "null".into()
    }
}

fn fmt_f64(v: Option<f64>) -> String {
    match v {
        Some(x) if x.is_finite() => format!("{x:.8e}"),
        _ => "null".into(),
    }
}

fn fmt_opt(v: Option<f64>) -> String {
    match v {
        Some(x) if x.is_finite() => format!("{x:.4e}"),
        _ => "null".into(),
    }
}

fn json_str(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if c.is_control() => o.push_str(&format!("\\u{:04x}", u32::from(c))),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

#[cfg(test)]
mod tests {
    #[test]
    fn sigkill_is_a_thrml_limit_and_not_a_result() {
        assert_eq!(super::limit_from_exit(Some(137), None), Some("oom"));
        assert_eq!(super::limit_from_exit(None, Some(9)), Some("oom"));
        assert_eq!(super::limit_from_exit(Some(1), None), None);
        assert!(super::thrml_oom_text(
            "jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate"
        ));
        assert!(super::thrml_oom_text("CUDA_ERROR_OUT_OF_MEMORY"));
        assert!(!super::thrml_oom_text("THRML runner exited exit status: 1"));
        assert_eq!(super::limit_from_exit(Some(0), Some(15)), None);
        let text =
            "META thrml=0.1.4\nLIMIT grid_scale_ising_4096 oom\nLIMIT grid_scale_ising_2048 oom\n";
        let (meta, map, limits) = super::parse_thrml(text).expect("limit lines parse");
        assert!(meta.contains("0.1.4"));
        assert!(map.is_empty());
        assert_eq!(
            limits.get("grid_scale_ising_4096").map(String::as_str),
            Some("oom")
        );
        assert_eq!(
            limits.get("grid_scale_ising_2048").map(String::as_str),
            Some("oom")
        );
        let reason = super::thrml_limit_reason("oom");
        assert!(reason.starts_with("THRML_LIMIT oom"));
        assert!(!super::parity_blocks_exit(&[(
            "thrml".into(),
            false,
            reason
        )]));
        assert!(super::parity_blocks_exit(&[(
            "thrml".into(),
            false,
            "per-site gap is past the line".into()
        )]));
    }

    #[test]
    fn grid_scale_flags_name_what_they_refuse() {
        let err = match super::Args::parse(
            ["--out", "out", "--scale", "--grid-scale"]
                .into_iter()
                .map(str::to_string),
        ) {
            Err(err) => err,
            Ok(_) => panic!("both workload sets were accepted"),
        };
        assert!(
            err.contains("--scale") && err.contains("--grid-scale"),
            "{err}"
        );
        let err = match super::Args::parse(
            ["--out", "out", "--profile"]
                .into_iter()
                .map(str::to_string),
        ) {
            Err(err) => err,
            Ok(_) => panic!("profile alone was accepted"),
        };
        assert!(err.contains("--profile"), "{err}");
        let err = match super::Args::parse(
            ["--out", "out", "--grid-scale", "--lever", "cuda"]
                .into_iter()
                .map(str::to_string),
        ) {
            Err(err) => err,
            Ok(_) => panic!("unknown lever was accepted"),
        };
        assert!(err.contains("cuda"), "{err}");
        let err = match super::Args::parse(
            ["--out", "out", "--grid-scale", "--sides", "7"]
                .into_iter()
                .map(str::to_string),
        ) {
            Err(err) => err,
            Ok(_) => panic!("side 7 was accepted"),
        };
        assert!(err.contains('7'), "{err}");
        assert!(
            super::Args::parse(
                [
                    "--out",
                    "out",
                    "--grid-scale",
                    "--profile",
                    "--sides",
                    "8,64",
                    "--skip-thrml"
                ]
                .into_iter()
                .map(str::to_string),
            )
            .is_ok()
        );
        let err = match super::Args::parse(
            ["--out", "out", "--grid-scale"]
                .into_iter()
                .map(str::to_string),
        ) {
            Err(err) => err,
            Ok(_) => panic!("grid-scale energy without --cpus was accepted"),
        };
        assert!(err.contains("--cpus"), "{err}");
        assert!(
            super::Args::parse(
                ["--out", "out", "--grid-scale", "--cpus", "2,5-6"]
                    .into_iter()
                    .map(str::to_string),
            )
            .is_ok()
        );
    }

    #[test]
    fn quiet_gate_refuses_to_share_a_run_with_contended_mode() {
        let Err(err) = super::Args::parse(
            ["--out", "out", "--quiet-gate", "--contended"]
                .into_iter()
                .map(str::to_string),
        ) else {
            panic!("--quiet-gate and --contended were accepted together");
        };
        assert!(err.contains("--quiet-gate"), "{err}");
        assert!(err.contains("--contended"), "{err}");
        assert!(
            super::Args::parse(
                ["--out", "out", "--quiet-gate"]
                    .into_iter()
                    .map(str::to_string)
            )
            .is_ok()
        );
    }

    #[test]
    fn jax_cuda_is_refused_when_the_only_device_is_cpu() {
        let script =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("python/test_platform.py");
        let out = std::process::Command::new("python3")
            .arg(&script)
            .output()
            .expect("python3");
        assert!(
            out.status.success(),
            "platform choice failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    #[test]
    fn the_harness_does_not_name_a_privileged_power_tool() {
        // The macOS path is macmon, inside ferrotherm-meter. These files are the harness.
        // Naming the forbidden tools here would make the prohibition a comment that can drift
        // into a call; the strings live only in this assertion.
        let forbidden = ["powermetrics", "sudo"];
        let files: [(&str, &str, usize); 4] = [
            ("energy.rs", include_str!("energy.rs"), 0),
            ("ft.rs", include_str!("ft.rs"), 0),
            // This file names the tools once, in the list above. Any second hit is a use.
            ("main.rs", include_str!("main.rs"), 1),
            ("runner.py", include_str!("../python/runner.py"), 0),
        ];
        for (name, src, allowed) in files {
            for word in forbidden {
                let hits = src.matches(word).count();
                assert!(hits <= allowed, "{name} contains {word} {hits} times");
            }
        }
    }

    #[test]
    fn projected_prices_are_not_a_field_the_report_can_fill_from_a_table() {
        let j = crate::energy::Joules::unavailable(1.0, "none", "test");
        let text = super::joules_json(&j);
        assert!(text.contains("\"projected_prices_applied\": false"));
        assert!(!text.contains("Z1_SPICE"));
        assert!(!text.contains("7.09e-15"));
    }
}
