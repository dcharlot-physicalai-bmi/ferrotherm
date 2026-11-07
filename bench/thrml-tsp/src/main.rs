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
mod score;
mod spec;

use energy::{
    energy_rank, grade_for, poll_quiet, read_cpu_snap, snap_other_cpu, trial_joules, wait_for_quiet_host,
    EnergyTrial, GradeSummary, Joules, Probe, GRADE_CONTENDED, MIN_ENERGY_WINDOW_S,
};
use ferrotherm_meter::Baseline;
use std::io::{BufRead, BufReader, Write};
use ft::Chain;
use score::{mean_ci_95, Score};
use spec::Workload;
use std::fs;
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
                 [--python PY] [--skip-thrml] [--contended] [--scale] [--device auto|cpu|cuda]"
            );
            std::process::exit(2);
        }
    };
    let mut work = spec::canonical(args.trials, args.warmup, args.draws);
    if args.scale {
        work.extend(spec::scaling(args.trials, args.warmup, args.draws));
    }
    let spec_text = spec::write_spec(&work);
    fs::create_dir_all(&args.out).unwrap_or_else(|e| panic!("create {}: {e}", args.out.display()));
    let traces = args.out.join("traces");
    fs::create_dir_all(&traces).unwrap_or_else(|e| panic!("create {}: {e}", traces.display()));
    fs::write(args.out.join("spec.txt"), &spec_text).expect("write spec");

    let mut probe = Probe::detect();
    eprintln!("wall: {}", probe.wall_note);
    eprintln!("rapl: {}", probe.rapl_note);
    eprintln!("perf: {}", probe.perf_note);
    eprintln!("nvml: {}", probe.nvml_note);
    eprintln!("load1: {:?} quiet: {}", probe.load1, probe.quiet);

    let mut body = String::new();
    let mut thrml_ok = args.skip_thrml;
    let mut thrml_note = if args.skip_thrml {
        "THRML runner skipped on request".to_string()
    } else {
        String::new()
    };
    let thrml_raw = if args.skip_thrml {
        None
    } else {
        match run_thrml(&args.python, &args.runner, &args.out.join("spec.txt"), &args.out.join("thrml_raw.txt"), &args.device) {
            Ok(()) => {
                thrml_ok = true;
                Some(fs::read_to_string(args.out.join("thrml_raw.txt")).unwrap_or_default())
            }
            Err(e) => {
                thrml_note = e;
                thrml_ok = false;
                None
            }
        }
    };
    let thrml_chains = thrml_raw.as_deref().map(parse_thrml).transpose();
    let (thrml_meta, thrml_by_id) = match thrml_chains {
        Ok(Some((meta, map))) => (meta, Some(map)),
        Ok(None) => (String::new(), None),
        Err(e) => {
            thrml_note = e;
            thrml_ok = false;
            (String::new(), None)
        }
    };

    for w in &work {
        eprintln!("ferrotherm {}", w.id);
        let mut ft_trials = Vec::new();
        for t in 0..w.trials {
            let seed = w.seed0 + t as u64;
            let chain = ft::run(w, seed).unwrap_or_else(|e| panic!("{} seed {seed}: {e}", w.id));
            if t == 0 {
                write_trace(&traces, w.id, "ferrotherm", &chain);
            }
            eprintln!(
                "  seed {seed} tv {:.4} tau {:.3} ess/s {}",
                chain.score.tv,
                chain.score.tau_int_energy,
                fmt_opt(chain.score.ess_per_second)
            );
            ft_trials.push(chain);
        }
        let thrml_trials = match &thrml_by_id {
            Some(map) => match map.get(w.id) {
                Some(raws) => Some(score_thrml(w, raws)),
                None => {
                    thrml_note = format!("THRML raw output has no workload {}", w.id);
                    thrml_ok = false;
                    None
                }
            },
            None => None,
        };
        if let Some(trials) = &thrml_trials
            && let Some(first) = trials.first()
        {
            write_trace(&traces, w.id, "thrml", first);
        }
        body.push_str(&workload_json(w, &ft_trials, thrml_trials.as_deref()));
        body.push_str(",\n");
    }
    if body.ends_with(",\n") {
        body.truncate(body.len() - 2);
        body.push('\n');
    }

    let energy = energy_section(
        &mut probe,
        &work,
        args.contended,
        args.trials,
        &args.python,
        &args.runner,
        &args.out.join("spec.txt"),
        !args.skip_thrml && thrml_ok,
        &args.device,
    );
    let report = format!(
        "{{\n  \"schema\": \"ferrotherm-thrml-bench/1\",\n  \"provenance\": {prov},\n  \
         \"energy\": {energy},\n  \"thrml_meta\": {meta},\n  \"thrml_error\": {err},\n  \
         \"workloads\": [\n{body}  ]\n}}\n",
        prov = provenance_json(&probe),
        meta = json_str(&thrml_meta),
        err = if thrml_note.is_empty() { "null".into() } else { json_str(&thrml_note) },
    );
    fs::write(args.out.join("report.json"), &report).expect("write report");
    eprintln!("wrote {}", args.out.join("report.json").display());
    if !thrml_ok {
        std::process::exit(1);
    }
}

struct Args {
    out: PathBuf,
    trials: usize,
    warmup: usize,
    draws: usize,
    python: PathBuf,
    runner: PathBuf,
    skip_thrml: bool,
    contended: bool,
    scale: bool,
    device: String,
}

impl Args {
    fn parse(args: impl Iterator<Item = String>) -> Result<Args, String> {
        let mut out = None;
        let mut trials = spec::TRIALS;
        let mut warmup = spec::WARMUP;
        let mut draws = spec::DRAWS;
        let mut python = std::env::var("THRML_PYTHON").unwrap_or_else(|_| "python3".into());
        let mut skip = false;
        let mut contended = false;
        let mut scale = false;
        let mut device = "auto".to_string();
        let mut it = args;
        while let Some(a) = it.next() {
            match a.as_str() {
                "--out" => out = Some(need(&mut it, "--out")?),
                "--trials" => trials = parse_usize(&need(&mut it, "--trials")?, "--trials")?,
                "--warmup" => warmup = parse_usize(&need(&mut it, "--warmup")?, "--warmup")?,
                "--draws" => draws = parse_usize(&need(&mut it, "--draws")?, "--draws")?,
                "--python" => python = need(&mut it, "--python")?,
                "--skip-thrml" => skip = true,
                "--contended" => contended = true,
                "--scale" => scale = true,
                "--device" => device = need(&mut it, "--device")?,
                "--help" | "-h" => return Err("help".into()),
                other => return Err(format!("unknown argument {other}")),
            }
        }
        let out = out.ok_or("missing --out")?;
        if trials < 2 {
            return Err("--trials must be at least 2 so a confidence interval exists".into());
        }
        if draws < 64 {
            return Err("--draws must be at least 64; tau_int is NaN under 16 and noisy near it".into());
        }
        if !matches!(device.as_str(), "auto" | "cpu" | "cuda") {
            return Err(format!("--device wants auto, cpu, or cuda, got {device}"));
        }
        Ok(Args {
            out: PathBuf::from(out),
            trials,
            warmup,
            draws,
            python: PathBuf::from(python),
            runner: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python/runner.py"),
            skip_thrml: skip,
            contended,
            scale,
            device,
        })
    }
}

fn need(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    it.next().ok_or_else(|| format!("{flag} needs a value"))
}

fn parse_usize(s: &str, flag: &str) -> Result<usize, String> {
    s.parse().map_err(|_| format!("{flag} wants an integer, got {s}"))
}

fn run_thrml(python: &Path, runner: &Path, spec: &Path, out: &Path, device: &str) -> Result<(), String> {
    eprintln!("THRML via {} {} --device {device}", python.display(), runner.display());
    let status = Command::new(python)
        .arg(runner)
        .arg("--spec")
        .arg(spec)
        .arg("--out")
        .arg(out)
        .arg("--device")
        .arg(device)
        .status()
        .map_err(|e| format!("could not start {}: {e}", python.display()))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("THRML runner exited {status}"))
    }
}

struct RawChain {
    seed: u64,
    seconds: f64,
    jit_seconds: f64,
    states_spin: Vec<Vec<i8>>,
    states_potts: Vec<Vec<u8>>,
}

fn parse_thrml(text: &str) -> Result<(String, std::collections::BTreeMap<String, Vec<RawChain>>), String> {
    let mut meta = String::new();
    let mut map: std::collections::BTreeMap<String, Vec<RawChain>> = std::collections::BTreeMap::new();
    let mut cur: Option<RawChainBuild> = None;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("META ") {
            meta = rest.to_string();
        } else if let Some(rest) = line.strip_prefix("WORK ") {
            if cur.is_some() {
                return Err("WORK inside an open WORK".into());
            }
            cur = Some(RawChainBuild::parse(rest)?);
        } else if let Some(rest) = line.strip_prefix("STATE ") {
            let c = cur.as_mut().ok_or("STATE outside WORK")?;
            c.push_state(rest)?;
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
    Ok((meta, map))
}

struct RawChainBuild {
    id: String,
    seed: u64,
    kind: String,
    seconds: f64,
    jit_seconds: f64,
    rows: Vec<Vec<i64>>,
}

impl RawChainBuild {
    fn parse(rest: &str) -> Result<Self, String> {
        let mut id = None;
        let mut seed = None;
        let mut kind = None;
        let mut seconds = None;
        let mut jit = None;
        for part in rest.split_whitespace() {
            let (k, v) = part.split_once('=').ok_or_else(|| format!("bad field {part}"))?;
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
        })
    }

    fn push_state(&mut self, rest: &str) -> Result<(), String> {
        let mut row = Vec::new();
        for tok in rest.split_whitespace() {
            row.push(tok.parse::<i64>().map_err(|_| format!("bad state token {tok}"))?);
        }
        self.rows.push(row);
        Ok(())
    }

    fn finish(self) -> Result<RawChain, String> {
        if self.rows.is_empty() {
            return Err(format!("{} produced no states", self.id));
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
        Ok(RawChain { seed: self.seed, seconds: self.seconds, jit_seconds: self.jit_seconds, states_spin: spins, states_potts: potts })
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
    // Re-score with the ferrotherm energy and the ferrotherm oracle. The external
    // runner contributed states and a wall-clock; it did not contribute the metric.
    ft::score_observed(w, raw.seconds, Some(raw.jit_seconds), &raw.states_spin, &raw.states_potts)
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

fn workload_json(w: &Workload, ft_trials: &[Chain], thrml: Option<&[Chain]>) -> String {
    let mut s = String::new();
    s.push_str("    {\n");
    s.push_str(&format!("      \"id\": {},\n", json_str(w.id)));
    s.push_str(&format!("      \"kind\": {},\n", json_str(w.kind.as_str())));
    s.push_str(&format!("      \"source\": {},\n", json_str(w.source)));
    s.push_str(&format!("      \"beta\": {:.16},\n", w.beta));
    s.push_str(&format!("      \"n\": {},\n", w.n));
    s.push_str(&format!("      \"q\": {},\n", w.q));
    s.push_str(&format!("      \"warmup\": {},\n", w.warmup));
    s.push_str(&format!("      \"draws\": {},\n", w.draws));
    s.push_str(&format!("      \"thin\": {},\n", w.thin));
    s.push_str(&format!("      \"oracle\": {},\n", json_str("exact enumeration of the Boltzmann distribution")));
    s.push_str("      \"ferrotherm\": ");
    s.push_str(&side_json(ft_trials, false));
    s.push_str(",\n      \"thrml\": ");
    match thrml {
        Some(t) => s.push_str(&side_json(t, true)),
        None => s.push_str("null"),
    }
    s.push_str("\n    }");
    s
}

fn side_json(trials: &[Chain], external: bool) -> String {
    let scores: Vec<&Score> = trials.iter().map(|c| &c.score).collect();
    let mut s = String::from("{\n");
    s.push_str("        \"summary\": ");
    s.push_str(&summary_json(&scores));
    s.push_str(",\n        \"trials\": [\n");
    for (i, c) in trials.iter().enumerate() {
        s.push_str(&trial_json(c, external));
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
    let ess_s: Vec<f64> = scores.iter().map(|s| s.ess_per_second.unwrap_or(f64::NAN)).collect();
    let de: Vec<f64> = scores.iter().map(|s| s.energy_abs_err).collect();
    let mag: Vec<f64> = scores.iter().map(|s| s.magnetization_mae.unwrap_or(f64::NAN)).collect();
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
        Some((mean, lo, hi, n)) => format!(
            "{{\"mean\": {mean:.8e}, \"lo95\": {lo:.8e}, \"hi95\": {hi:.8e}, \"n\": {n}}}"
        ),
        None => "null".into(),
    }
}

fn trial_json(c: &Chain, external: bool) -> String {
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
         \"ess_per_second\": {ess_s}, \"mean_energy\": {:.8e}, \"exact_mean_energy\": {:.8e}, \
         \"energy_abs_err\": {:.8e}, \"magnetization_mae\": {mag}, \
         \"gibbs_updates\": {}, \"reads\": {}, \
         \"measured_joules_per_ess\": null}}",
        sc.draws,
        sc.seconds,
        fmt_finite(sc.tv),
        fmt_finite(sc.kl_empirical_exact),
        sc.missed_states,
        sc.tau_int_energy,
        sc.ess_energy,
        sc.mean_energy,
        sc.exact_mean_energy,
        sc.energy_abs_err,
        c.updates,
        c.reads,
    )
}

fn energy_section(
    probe: &mut Probe,
    work: &[Workload],
    contended: bool,
    trials: usize,
    python: &Path,
    runner: &Path,
    spec: &Path,
    thrml_ok: bool,
    device: &str,
) -> String {
    let poll = if contended { poll_quiet(&[ferrotherm::host::load_average()]) } else { wait_for_quiet_host() };
    let quiet = poll.outcome == "quiet";
    let grade = grade_for(contended, quiet);
    let has_instrument = probe.rapl.is_some()
        || probe.perf.is_some()
        || probe.wall.is_some()
        || probe.nvml.is_some()
        || probe.tegra.is_some();
    if grade == "load_gated" {
        let gated = energy::load_gated(
            "No joules. The quiet-host gate is unchanged (load average above 2). After the bounded \
             wait the host was still contended, and this run did not opt into contended-host mode, \
             so no energy figure was taken. A contended figure would be a different grade and is \
             not implied here.",
        );
        return energy_json(
            probe,
            &poll,
            grade,
            gated.evidence,
            None,
            &[],
            &gated.note,
        );
    }
    if trials < 5 {
        return energy_json(
            probe,
            &poll,
            grade,
            "too_few_trials",
            None,
            &[],
            "No joules. The energy head-to-head needs at least 5 interleaved trials so a 95% interval exists.",
        );
    }
    if !has_instrument {
        let missing = Joules::unavailable(
            0.0,
            "none",
            "No joules were obtained. No macmon, INA3221, tegrastats VDD_IN, RAPL powercap, \
             perf power/energy-pkg, or NVML counter opened on this host. Nothing was filled in \
             from a device-price table.",
        );
        return energy_json(probe, &poll, grade, missing.evidence, None, &[], &missing.note);
    }
    let idle = match take_idle(probe, contended) {
        Ok(b) => b,
        Err(e) => {
            return energy_json(probe, &poll, grade, "unavailable", None, &[], &e);
        }
    };
    let gpu_idle = probe.nvml.as_ref().and_then(|n| nvml_idle(n).ok());
    let mut session = if thrml_ok {
        ThrmlServe::start(python, runner, spec, device).ok()
    } else {
        None
    };
    let mut rows = Vec::new();
    for w in work {
        let mut ft_n = 0usize;
        let mut th_n = 0usize;
        for side in energy::ab_schedule(trials) {
            if side == "thrml" && session.is_none() {
                continue;
            }
            let t = if side == "ferrotherm" {
                let i = ft_n;
                ft_n += 1;
                i
            } else {
                let i = th_n;
                th_n += 1;
                i
            };
            let seed = w.seed0 + t as u64 + 10_000;
            let before = read_cpu_snap();
            let measured = measure_side(probe, side, w, seed, session.as_mut());
            let after = read_cpu_snap();
            let rank = energy_rank(measured.instrument, &probe.hostname);
            let joules = trial_joules(
                measured.gross,
                idle.0,
                idle.1,
                measured.seconds,
                measured.instrument,
                measured.scope,
                grade,
                measured.counter_updates,
            );
            rows.push(EnergyTrial::from_parts(
                w.id,
                side,
                t,
                measured.repetitions,
                measured.counter_updates,
                measured.window_seconds,
                joules,
                measured.ess,
                after.load1,
                snap_other_cpu(&before, &after),
                grade,
                rank,
            ));
            if measured.instrument != "nvml"
                && let Some((gw, gs)) = gpu_idle
            {
                let gpu = measure_counter(probe, side, w, seed, session.as_mut(), CounterKind::Nvml);
                let gpu_rank = energy_rank(gpu.instrument, &probe.hostname);
                let gpu_joules = trial_joules(
                    gpu.gross,
                    gw,
                    gs,
                    gpu.seconds,
                    gpu.instrument,
                    gpu.scope,
                    grade,
                    gpu.counter_updates,
                );
                let gpu_after = read_cpu_snap();
                rows.push(EnergyTrial::from_parts(
                    w.id,
                    side,
                    t,
                    gpu.repetitions,
                    gpu.counter_updates,
                    gpu.window_seconds,
                    gpu_joules,
                    gpu.ess,
                    gpu_after.load1,
                    None,
                    grade,
                    gpu_rank,
                ));
            }
        }
    }
    energy_json(probe, &poll, grade, "measured", Some(idle), &rows, &energy_note(grade))
}

fn energy_note(grade: &str) -> String {
    if grade == GRADE_CONTENDED {
        "Evidence grade contended host. Each trial repeats the sampler for at least 2 s and \
         reports joules per repetition. Attribution needs 8 counter updates in that window. \
         Summaries are per workload. RAPL package and macmon whole-system are not combined. \
         These figures are not quiet-host evidence. load1 and other-process CPU are on every trial. \
         RAPL on framework-128gb-0000 reads package energy_uj without root and is the primary \
         instrument. macmon is secondary."
            .into()
    } else {
        "Evidence grade quiet. The host cleared the unchanged load gate (at or under 2). \
         Each trial repeats the sampler for at least 2 s and reports joules per repetition. \
         Attribution needs 8 counter updates. Summaries are per workload. RAPL package and \
         macmon whole-system are not combined. RAPL on framework-128gb-0000 reads package \
         energy_uj without root and is primary; macmon is secondary."
            .into()
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
    energy::watts_from_windows(&windows).ok_or_else(|| "idle counter produced fewer than two windows".into())
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
        return Ok((energy::counter_delta_j(a, b, u64::MAX as f64 / 1000.0), t.elapsed().as_secs_f64()));
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
    scope: &'static str,
}

fn measure_side(
    probe: &mut Probe,
    side: &str,
    w: &Workload,
    seed: u64,
    session: Option<&mut ThrmlServe>,
) -> SideMeasure {
    if probe.wall.is_some() {
        return measure_wall(probe, side, w, seed, session);
    }
    if probe.tegra.is_some() {
        return measure_tegra(probe, side, w, seed, session);
    }
    if probe.rapl.is_some() {
        return measure_counter(probe, side, w, seed, session, CounterKind::Rapl);
    }
    if probe.perf.is_some() {
        return measure_counter(probe, side, w, seed, session, CounterKind::Perf);
    }
    measure_counter(probe, side, w, seed, session, CounterKind::Nvml)
}

fn measure_wall(
    probe: &mut Probe,
    side: &str,
    w: &Workload,
    seed: u64,
    mut session: Option<&mut ThrmlServe>,
) -> SideMeasure {
    let wall = probe.wall.as_mut().expect("wall was checked");
    let backend = wall.backend();
    let (instrument, scope) = if backend == "macmon" { ("macmon", "whole-system") } else { ("ina3221", "board") };
    let mut reps = 0u32;
    let mut ess_sum = 0.0;
    let integrated = wall.integrate(Baseline { watts: 0.0, sigma: 0.0, samples: 0, load1: None }, || {
        let (r, e) = repeat_until_window(seed, |seed_i| plan_ess(side, w, seed_i, &mut session));
        reps = r;
        ess_sum = e;
    });
    // The zero baseline above is only so integrate returns a Run. Gross is mean watts
    // times seconds. Incremental is applied by the caller against the real idle.
    // Joules and ESS below are per repetition.
    match integrated {
        Ok((run, ())) => finish_window(
            ess_sum,
            run.joules_total,
            run.seconds,
            reps,
            u32::try_from(run.samples).unwrap_or(u32::MAX),
            instrument,
            scope,
        ),
        Err(_) => SideMeasure {
            ess: f64::NAN,
            gross: f64::NAN,
            seconds: 0.0,
            window_seconds: 0.0,
            repetitions: reps,
            counter_updates: 0,
            instrument,
            scope,
        },
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
    mut session: Option<&mut ThrmlServe>,
) -> SideMeasure {
    let t0 = std::time::Instant::now();
    let (reps, ess_sum) = repeat_until_window(seed, |seed_i| plan_ess(side, w, seed_i, &mut session));
    let seconds = t0.elapsed().as_secs_f64();
    let watts = probe.tegra.as_ref().map(|t| t.since(t0)).unwrap_or_default();
    let updates = u32::try_from(watts.len()).unwrap_or(u32::MAX);
    let mean = if watts.is_empty() { f64::NAN } else { watts.iter().sum::<f64>() / watts.len() as f64 };
    finish_window(ess_sum, mean * seconds, seconds, reps, updates, "tegrastats", "board")
}

fn nvml_idle(nvml: &energy::Nvml) -> Result<(f64, f64), String> {
    let mut windows = Vec::new();
    for _ in 0..6 {
        let a = nvml.millijoules()?;
        let t0 = std::time::Instant::now();
        std::thread::sleep(std::time::Duration::from_secs_f64(0.25));
        let b = nvml.millijoules()?;
        windows.push((energy::mj_delta_j(a, b), t0.elapsed().as_secs_f64()));
    }
    energy::watts_from_windows(&windows).ok_or_else(|| "NVML idle produced fewer than two windows".into())
}

fn measure_counter(
    probe: &mut Probe,
    side: &str,
    w: &Workload,
    seed: u64,
    mut session: Option<&mut ThrmlServe>,
    kind: CounterKind,
) -> SideMeasure {
    let opened = match kind {
        CounterKind::Rapl => {
            let rapl = probe.rapl.as_ref().expect("rapl was checked");
            let scope = if rapl.kind == "psys" { "psys" } else { "package" };
            let max_j = rapl.range_j();
            let path = rapl.counter_path().to_path_buf();
            OpenedCounter {
                instrument: "rapl",
                scope,
                gross: Gross::Joules { max_j: Some(max_j) },
                read: Box::new(move || {
                    let text = std::fs::read_to_string(path.join("energy_uj")).ok()?;
                    let uj: u64 = text.trim().parse().ok()?;
                    Some(uj as f64 / 1e6)
                }),
            }
        }
        CounterKind::Perf => {
            let (fd, scale) = probe.perf.as_ref().expect("perf was checked").fd_and_scale();
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
    let (readings, window_s, (reps, ess_sum)) = energy::sample_during(opened.read, || {
        repeat_until_window(seed, |seed_i| plan_ess(side, w, seed_i, &mut session))
    });
    let gross_window = match opened.gross {
        Gross::Joules { max_j } => energy::window_gross(&readings, max_j),
        Gross::Millijoules => energy::mj_window_j(&readings),
    };
    finish_window(
        ess_sum,
        gross_window,
        window_s,
        reps,
        energy::count_counter_updates(&readings),
        opened.instrument,
        opened.scope,
    )
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
    window_seconds: f64,
    repetitions: u32,
    counter_updates: u32,
    instrument: &'static str,
    scope: &'static str,
) -> SideMeasure {
    let (gross, seconds) = energy::per_repetition(gross_window, window_seconds, repetitions);
    let ess = energy::per_repetition(ess_sum, window_seconds, repetitions).0;
    SideMeasure { ess, gross, seconds, window_seconds, repetitions, counter_updates, instrument, scope }
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

fn plan_ess(side: &str, w: &Workload, seed: u64, session: &mut Option<&mut ThrmlServe>) -> f64 {
    one_plan(side, w, seed, session.as_deref_mut())
}

fn one_plan(side: &str, w: &Workload, seed: u64, session: Option<&mut ThrmlServe>) -> f64 {
    if side == "ferrotherm" {
        ft::run(w, seed).map_or(f64::NAN, |c| c.score.ess_energy)
    } else if let Some(s) = session {
        s.run(w, seed).map_or(f64::NAN, |c| c.score.ess_energy)
    } else {
        f64::NAN
    }
}

fn energy_json(
    probe: &Probe,
    poll: &energy::QuietPoll,
    grade: &str,
    outcome: &str,
    idle: Option<(f64, f64)>,
    rows: &[EnergyTrial],
    note: &str,
) -> String {
    let summaries = energy::summarise_workloads(rows);
    format!(
        "{{\n    \"evidence_grade\": {},\n    \"outcome\": {},\n    \"quiet_threshold\": {},\n    \
         \"load_samples\": {},\n    \"rank_policy\": {},\n    \"order\": {},\n    \
         \"window_s\": {},\n    \"min_counter_updates\": {},\n    \
         \"backends\": {{\"macmon_or_ina\": {}, \"tegrastats\": {}, \"rapl\": {}, \"perf\": {}, \"nvml\": {}}},\n    \
         \"hostname\": {},\n    \"macos_policy\": {},\n    \"accepted_load\": {},\n    \
         \"idle_watts\": {},\n    \"idle_sigma\": {},\n    \
         \"summaries\": [\n{}    ],\n    \"trials\": [\n{}    ],\n    \
         \"projected_prices_applied\": false,\n    \"note\": {}\n  }}",
        json_str(grade),
        json_str(outcome),
        poll.threshold,
        loads_json(&poll.samples),
        json_str(
            "primary = RAPL package on framework-128gb-0000, read without root; macmon whole-system = secondary; scopes are not combined",
        ),
        json_str("A/B/A/B... ferrotherm, THRML, ferrotherm, ...; joules are per repetition of a >=2 s window"),
        MIN_ENERGY_WINDOW_S,
        energy::MIN_COUNTER_UPDATES,
        json_str(&probe.wall_note),
        json_str(&probe.tegra_note),
        json_str(&probe.rapl_note),
        json_str(&probe.perf_note),
        json_str(&probe.nvml_note),
        json_str(&probe.hostname),
        json_str(probe.macos_policy),
        fmt_f64(poll.accepted),
        fmt_f64(idle.map(|p| p.0)),
        fmt_f64(idle.map(|p| p.1)),
        summaries.iter().map(grade_summary_json).collect::<Vec<_>>().join(",\n"),
        rows.iter().map(trial_line).collect::<Vec<_>>().join(",\n"),
        json_str(note),
    )
}

fn loads_json(loads: &[f64]) -> String {
    let parts: Vec<String> = loads.iter().map(|l| if l.is_finite() { format!("{l:.4}") } else { "null".into() }).collect();
    format!("[{}]", parts.join(", "))
}

fn grade_summary_json(s: &GradeSummary) -> String {
    format!(
        "      {{\"workload\": {}, \"side\": {}, \"instrument\": {}, \"scope\": {}, \"grade\": {}, \
         \"n\": {}, \"gross_j\": {}, \"incremental_j\": {}, \"joules_per_ess\": {}}}",
        json_str(s.workload),
        json_str(s.side),
        json_str(&s.instrument),
        json_str(&s.scope),
        json_str(s.grade),
        s.n,
        ci_opt(s.gross),
        ci_opt(s.incremental),
        ci_opt(s.joules_per_ess),
    )
}

fn ci_opt(v: Option<(f64, f64, f64, usize)>) -> String {
    match v {
        Some((mean, lo, hi, n)) => format!(
            "{{\"mean\": {mean:.8e}, \"lo95\": {lo:.8e}, \"hi95\": {hi:.8e}, \"n\": {n}}}"
        ),
        None => "null".into(),
    }
}

fn trial_line(t: &EnergyTrial) -> String {
    format!(
        "      {{\"workload\": {}, \"side\": {}, \"trial\": {}, \"repetitions\": {}, \
         \"counter_updates\": {}, \"window_seconds\": {:.6}, \"grade\": {}, \"rank\": {}, \
         \"load1\": {}, \"other_process_cpu_s\": {}, \"ess\": {}, \"joules_per_ess\": {}, \
         \"joules\": {}}}",
        json_str(t.workload),
        json_str(t.side),
        t.trial,
        t.repetitions,
        t.counter_updates,
        t.window_seconds,
        json_str(t.grade),
        json_str(t.rank),
        fmt_f64(t.load1),
        fmt_f64(t.other_cpu_s),
        fmt_f64(Some(t.ess)),
        fmt_f64(t.joules_per_ess),
        joules_json(&t.joules),
    )
}

struct ThrmlServe {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    out: BufReader<std::process::ChildStdout>,
}

impl ThrmlServe {
    fn start(python: &Path, runner: &Path, spec: &Path, device: &str) -> Result<ThrmlServe, String> {
        let mut child = Command::new(python)
            .arg(runner)
            .arg("--serve")
            .arg("--device")
            .arg(device)
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
        out.read_line(&mut line).map_err(|e| format!("THRML serve ready: {e}"))?;
        if !line.starts_with("READY") {
            return Err(format!("THRML serve said {line:?}"));
        }
        Ok(ThrmlServe { child, stdin, out })
    }

    fn run(&mut self, w: &Workload, seed: u64) -> Result<Chain, String> {
        writeln!(self.stdin, "RUN {} {seed}", w.id).map_err(|e| format!("THRML serve write: {e}"))?;
        self.stdin.flush().ok();
        let mut buf = String::new();
        loop {
            let mut line = String::new();
            let n = self.out.read_line(&mut line).map_err(|e| format!("THRML serve read: {e}"))?;
            if n == 0 {
                return Err("THRML serve closed".into());
            }
            buf.push_str(&line);
            if line.trim() == "END" {
                break;
            }
        }
        let (_, mut map) = parse_thrml(&buf)?;
        let raw = map.remove(w.id).and_then(|mut v| v.pop()).ok_or("THRML serve returned no chain")?;
        score_external(w, &raw)
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
        if j.projected_prices_applied { "true" } else { "false" },
    )
}

fn provenance_json(probe: &Probe) -> String {
    let rev = Command::new("git").args(["rev-parse", "HEAD"]).output().ok();
    let rev = rev.filter(|o| o.status.success()).map_or_else(
        || "unknown".into(),
        |o| String::from_utf8_lossy(&o.stdout).trim().to_string(),
    );
    let dirty = Command::new("git").args(["status", "--porcelain"]).output().ok();
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
        "{{\n    \"git_rev\": {},\n    \"git_dirty\": {},\n    \"host_cpu\": {},\n    \
         \"os\": {},\n    \"rustc\": {},\n    \"thrml_pin\": {},\n    \"thrml_wheel_sha256\": {},\n    \
         \"load1_at_start\": {},\n    \"what_this_machine_measured\": {},\n    \
         \"what_still_needs_fleet_hardware\": {}\n  }}",
        json_str(&rev),
        if dirty { "true" } else { "false" },
        json_str(&cpu),
        json_str(std::env::consts::OS),
        json_str(&rustc_version()),
        json_str("thrml==0.1.4"),
        json_str("6e2f38cecb562589d230ca063b5fcb5d2a6533201e37bb70c1f2dac4a63a0858"),
        match probe.load1 {
            Some(v) => format!("{v:.4}"),
            None => "null".into(),
        },
        json_str(&format!("wall: {}. nvml: {}.", probe.wall_note, probe.nvml_note)),
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
    Command::new("rustc").arg("--version").output().ok().filter(|o| o.status.success()).map_or_else(
        || "unknown".into(),
        |o| String::from_utf8_lossy(&o.stdout).trim().to_string(),
    )
}

fn fmt_finite(x: f64) -> String {
    if x.is_finite() { format!("{x:.8e}") } else { "null".into() }
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
    fn jax_cuda_is_refused_when_the_only_device_is_cpu() {
        let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("python/test_platform.py");
        let out = std::process::Command::new("python3").arg(&script).output().expect("python3");
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
