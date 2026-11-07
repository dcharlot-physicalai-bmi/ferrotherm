//! Head-to-head: ferrotherm and a pinned THRML runner, same workloads, one scorer.
//!
//! Accuracy is total variation and KL against an exact Boltzmann oracle, integrated
//! autocorrelation time and effective sample size on the energy trace, and the error
//! of the mean energy (and, for spins, the per-site magnetisation).
//!
//! Energy is a hardware counter or it is absent. Projected device prices are not read.

mod energy;
mod ft;
mod score;
mod spec;

use energy::{joules_per_ess, mj_delta_j, split_incremental, Joules, Probe};
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
                 [--python PY] [--skip-thrml]"
            );
            std::process::exit(2);
        }
    };
    let work = spec::canonical(args.trials, args.warmup, args.draws);
    let spec_text = spec::write_spec(&work);
    fs::create_dir_all(&args.out).unwrap_or_else(|e| panic!("create {}: {e}", args.out.display()));
    let traces = args.out.join("traces");
    fs::create_dir_all(&traces).unwrap_or_else(|e| panic!("create {}: {e}", traces.display()));
    fs::write(args.out.join("spec.txt"), &spec_text).expect("write spec");

    let mut probe = Probe::detect();
    eprintln!("wall: {}", probe.wall_note);
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
        match run_thrml(&args.python, &args.runner, &args.out.join("spec.txt"), &args.out.join("thrml_raw.txt")) {
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

    let energy = energy_section(&mut probe, &work);
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
}

impl Args {
    fn parse(args: impl Iterator<Item = String>) -> Result<Args, String> {
        let mut out = None;
        let mut trials = spec::TRIALS;
        let mut warmup = spec::WARMUP;
        let mut draws = spec::DRAWS;
        let mut python = std::env::var("THRML_PYTHON").unwrap_or_else(|_| "python3".into());
        let mut skip = false;
        let mut it = args;
        while let Some(a) = it.next() {
            match a.as_str() {
                "--out" => out = Some(need(&mut it, "--out")?),
                "--trials" => trials = parse_usize(&need(&mut it, "--trials")?, "--trials")?,
                "--warmup" => warmup = parse_usize(&need(&mut it, "--warmup")?, "--warmup")?,
                "--draws" => draws = parse_usize(&need(&mut it, "--draws")?, "--draws")?,
                "--python" => python = need(&mut it, "--python")?,
                "--skip-thrml" => skip = true,
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
        Ok(Args {
            out: PathBuf::from(out),
            trials,
            warmup,
            draws,
            python: PathBuf::from(python),
            runner: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python/runner.py"),
            skip_thrml: skip,
        })
    }
}

fn need(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    it.next().ok_or_else(|| format!("{flag} needs a value"))
}

fn parse_usize(s: &str, flag: &str) -> Result<usize, String> {
    s.parse().map_err(|_| format!("{flag} wants an integer, got {s}"))
}

fn run_thrml(python: &Path, runner: &Path, spec: &Path, out: &Path) -> Result<(), String> {
    eprintln!("THRML via {} {}", python.display(), runner.display());
    let status = Command::new(python)
        .arg(runner)
        .arg("--spec")
        .arg(spec)
        .arg("--out")
        .arg(out)
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
         \"tv\": {:.8e}, \"kl_empirical_exact\": {:.8e}, \"kl_exact_empirical\": {kl_rev}, \
         \"missed_states\": {}, \"tau_int_energy\": {:.8e}, \"ess_energy\": {:.8e}, \
         \"ess_per_second\": {ess_s}, \"mean_energy\": {:.8e}, \"exact_mean_energy\": {:.8e}, \
         \"energy_abs_err\": {:.8e}, \"magnetization_mae\": {mag}, \
         \"gibbs_updates\": {}, \"reads\": {}, \
         \"measured_joules_per_ess\": null}}",
        sc.draws,
        sc.seconds,
        sc.tv,
        sc.kl_empirical_exact,
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

fn energy_section(probe: &mut Probe, work: &[Workload]) -> String {
    // Accuracy chains of a few thousand sweeps on n <= 16 finish in milliseconds, under
    // the meter's minimum window. Ask the meter about a separate repeat that lasts at
    // least two seconds. If it refuses, the refusal is the result.
    let joules = match long_run(probe, &work[0]) {
        Ok(j) => j,
        Err(note) => Joules::unavailable(0.0, "none", &note),
    };
    // The long run is not the scored chain, so there is no ESS to divide by.
    // Calling the function keeps that refusal in the same place as a real division.
    let per_ess = joules_per_ess(&joules, f64::NAN);
    format!(
        "{{\n    \"wall_probe\": {},\n    \"nvml_probe\": {},\n    \"load1\": {},\n    \
         \"quiet\": {},\n    \"macos_policy\": {},\n    \"long_run\": {},\n    \
         \"joules_per_effective_sample\": {},\n    \
         \"projected_prices_applied\": false,\n    \
         \"note\": {}\n  }}",
        json_str(&probe.wall_note),
        json_str(&probe.nvml_note),
        match probe.load1 {
            Some(v) => format!("{v:.4}"),
            None => "null".into(),
        },
        if probe.quiet { "true" } else { "false" },
        json_str(probe.macos_policy),
        joules_json(&joules),
        fmt_f64(per_ess),
        json_str(
            "Per-trial joules are null: those chains are shorter than a power-counter window. \
             long_run is a separate repeat of the first workload, metered only if a counter is \
             present. Joules per effective sample are not computed from it, because that repeat \
             is not the chain whose ESS was scored, so joules_per_effective_sample stays null \
             even when long_run cleared the noise floor. Projected device prices, including \
             Z1_SPICE, are not an input.",
        ),
    )
}

fn long_run(probe: &mut Probe, w: &Workload) -> Result<Joules, String> {
    if probe.wall.is_none() && probe.nvml.is_none() {
        return Err("no RAPL, INA3221, macmon, or NVML counter on this machine".into());
    }
    if probe.wall.is_some() {
        let baseline = match energy::idle_baseline(probe) {
            Ok(b) => b,
            Err(e) => return Ok(Joules::unavailable(0.0, "wall", &e)),
        };
        let nvml_before = probe.nvml.as_ref().and_then(|n| n.joules().ok());
        let started = Instant::now();
        let mut repeats = 0u32;
        let measured = probe.wall.as_mut().expect("meter present").measure(baseline, || {
            let t0 = Instant::now();
            while t0.elapsed().as_secs_f64() < 2.0 {
                let _ = ft::run(w, 1);
                repeats += 1;
            }
        });
        let nvml_after = probe.nvml.as_ref().and_then(|n| n.joules().ok());
        let seconds = started.elapsed().as_secs_f64();
        return match measured {
            Ok(run) => {
                let (inc, ok) = split_incremental(run.joules_total, run.idle_watts, run.idle_sigma, run.seconds);
                // The meter already refused anything under its floor, so `ok` here is a second look
                // with the same 3-sigma rule. Prefer the meter's own incremental, which cleared it.
                let _ = (inc, ok, repeats);
                Ok(Joules {
                    gross: Some(run.joules_total),
                    incremental: Some(run.joules_above_idle),
                    idle_watts: Some(run.idle_watts),
                    idle_sigma: Some(run.idle_sigma),
                    seconds: run.seconds,
                    attributable: true,
                    evidence: "metered",
                    instrument: run.backend.to_string(),
                    scope: run.machine.clone(),
                    note: format!(
                        "separate >=2s repeat of {}, {repeats} calls; not the scored chain. \
                         NVML joules over a wider window: {}",
                        w.id,
                        nvml_pair(nvml_before, nvml_after)
                    ),
                    projected_prices_applied: false,
                })
            }
            Err(e) => Ok(Joules::unavailable(
                seconds,
                "wall",
                &format!("{e} NVML: {}", nvml_pair(nvml_before, nvml_after)),
            )),
        };
    }
    // NVML only. Diff the energy counter around a two-second repeat. Idle is a two-second
    // counter delta taken first, and only while the machine is quiet.
    if !probe.quiet {
        return Ok(Joules::unavailable(
            0.0,
            "nvml",
            "machine is not quiet; refusing an NVML idle baseline",
        ));
    }
    let idle_j = nvml_over(probe, 2.0)?;
    let idle_w = idle_j / 2.0;
    let run_j = {
        let t0 = Instant::now();
        let before = probe.nvml.as_ref().unwrap().millijoules()?;
        while t0.elapsed().as_secs_f64() < 2.0 {
            let _ = ft::run(w, 1);
        }
        let after = probe.nvml.as_ref().unwrap().millijoules()?;
        let seconds = t0.elapsed().as_secs_f64();
        (mj_delta_j(before, after), seconds)
    };
    let (inc, ok) = split_incremental(run_j.0, idle_w, 0.0, run_j.1);
    // sigma is unknown from two counter endpoints. Mark not attributable: a single idle
    // window has no wander to compare against. Gross is still the counter delta.
    let _ = ok;
    Ok(Joules {
        gross: Some(run_j.0),
        incremental: Some(inc),
        idle_watts: Some(idle_w),
        idle_sigma: None,
        seconds: run_j.1,
        attributable: false,
        evidence: "metered",
        instrument: "nvml".into(),
        scope: format!("sum of GPU energy counters ({})", probe.nvml_note),
        note: "gross is the NVML total-energy counter delta. Incremental subtracts one idle \
               window and is not attributable: that window was not repeated, so it has no sigma. \
               Not added to a CPU package counter, because no CPU counter was open."
            .into(),
        projected_prices_applied: false,
    })
}

fn nvml_over(probe: &Probe, seconds: f64) -> Result<f64, String> {
    let n = probe.nvml.as_ref().ok_or("no NVML")?;
    let a = n.millijoules()?;
    std::thread::sleep(std::time::Duration::from_secs_f64(seconds));
    let b = n.millijoules()?;
    Ok(mj_delta_j(a, b))
}

fn nvml_pair(before: Option<f64>, after: Option<f64>) -> String {
    match (before, after) {
        (Some(a), Some(b)) => format!("{:.6} J", b - a),
        _ => "not read".into(),
    }
}

fn joules_json(j: &Joules) -> String {
    format!(
        "{{\"gross_j\": {}, \"incremental_j\": {}, \"idle_watts\": {}, \"idle_sigma\": {}, \
         \"seconds\": {:.6}, \"attributable\": {}, \"evidence\": {}, \"instrument\": {}, \
         \"scope\": {}, \"note\": {}, \"projected_prices_applied\": {}}}\n",
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
