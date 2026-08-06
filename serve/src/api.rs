//! The operations, once, so that HTTP and MCP cannot drift apart.
//!
//! Every handler takes a JSON request and returns a JSON result or an error string. The transports
//! in `http` and `mcp` do nothing but move bytes and shape envelopes.

use crate::json::{parse, Json};
use ferrotherm::gibbs::Sampler;
use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::ising;
use ferrotherm::certify::{certify, Certificate};
use ferrotherm::ledger::{Ledger, Prices, Z1_SPICE};
use std::time::Instant;

/// Ceilings that keep one request from monopolising the process. They are advertised in
/// `capabilities` so a caller can size a job instead of discovering the wall by hitting it.
pub const MAX_NODES: usize = 4_000_000;
pub const MAX_NODE_UPDATES: u64 = 20_000_000_000;
/// Total spins retained across all certified draws, bounding memory at about 20 MB.
pub const RETAINED_SPIN_BUDGET: usize = 20_000_000;

// ---- graph construction -----------------------------------------------------------------------

/// Build a graph from either an explicit edge list or a named family.
///
/// Explicit: `{"n": 4, "couplings": [[0,1,1.0]], "biases": [[0,0.5]]}`
/// Family:   `{"builtin": "lattice2d", "l": 32, "j": 1.0}`
pub fn graph_from(v: &Json) -> Result<Graph, String> {
    if let Some(kind) = v.get("builtin").and_then(|b| b.as_str()) {
        return match kind {
            "ring" => {
                let n = req_usize(v, "n")?;
                bound_nodes(n)?;
                Ok(ising::ring(n, opt_f64(v, "j", 1.0), opt_f64(v, "h", 0.0)))
            }
            "lattice2d" => {
                let l = req_usize(v, "l")?;
                bound_nodes(l.saturating_mul(l))?;
                Ok(ising::lattice2d(l, opt_f64(v, "j", 1.0)))
            }
            other => Err(format!(
                "unknown builtin {other:?}; known families are \"ring\" and \"lattice2d\", \
                 or give an explicit {{\"n\":..,\"couplings\":[[i,j,J]]}}"
            )),
        };
    }

    let n = req_usize(v, "n").map_err(|_| {
        "graph needs either \"builtin\" (\"ring\" or \"lattice2d\") or an explicit \"n\" \
         with \"couplings\""
            .to_string()
    })?;
    bound_nodes(n)?;
    let mut b = GraphBuilder::new(n);

    if let Some(cs) = v.get("couplings") {
        let cs = cs.as_arr().ok_or("\"couplings\" must be an array of [i, j, J]")?;
        for (k, c) in cs.iter().enumerate() {
            let t = c.as_arr().ok_or_else(|| format!("coupling {k} must be [i, j, J]"))?;
            if t.len() != 3 {
                return Err(format!("coupling {k} must have 3 entries [i, j, J], got {}", t.len()));
            }
            let (i, j) = (idx(&t[0], n, k, "i")?, idx(&t[1], n, k, "j")?);
            if i == j {
                return Err(format!(
                    "coupling {k} connects node {i} to itself; a self-coupling is a bias, \
                     put it in \"biases\""
                ));
            }
            let w = t[2].as_f64().ok_or_else(|| format!("coupling {k}: J must be a number"))?;
            if !w.is_finite() {
                return Err(format!("coupling {k}: J must be finite"));
            }
            b.couple(i, j, w);
        }
    }
    if let Some(hs) = v.get("biases") {
        let hs = hs.as_arr().ok_or("\"biases\" must be an array of [i, h]")?;
        for (k, e) in hs.iter().enumerate() {
            let t = e.as_arr().ok_or_else(|| format!("bias {k} must be [i, h]"))?;
            if t.len() != 2 {
                return Err(format!("bias {k} must have 2 entries [i, h], got {}", t.len()));
            }
            let i = idx(&t[0], n, k, "i")?;
            let h = t[1].as_f64().ok_or_else(|| format!("bias {k}: h must be a number"))?;
            if !h.is_finite() {
                return Err(format!("bias {k}: h must be finite"));
            }
            b.bias(i, h);
        }
    }
    Ok(b.build())
}

fn bound_nodes(n: usize) -> Result<(), String> {
    if n == 0 {
        return Err("graph must have at least one node".into());
    }
    if n > MAX_NODES {
        return Err(format!("graph has {n} nodes, over the {MAX_NODES} ceiling for one request"));
    }
    Ok(())
}

fn idx(v: &Json, n: usize, k: usize, which: &str) -> Result<usize, String> {
    let i = v.as_usize().ok_or_else(|| format!("entry {k}: {which} must be a non-negative integer"))?;
    if i >= n {
        return Err(format!("entry {k}: {which} = {i} is out of range for a graph of {n} nodes"));
    }
    Ok(i)
}

fn req_usize(v: &Json, key: &str) -> Result<usize, String> {
    v.get(key)
        .and_then(|x| x.as_usize())
        .ok_or_else(|| format!("missing or non-integer field {key:?}"))
}

fn opt_f64(v: &Json, key: &str, dflt: f64) -> f64 {
    v.get(key).and_then(|x| x.as_f64()).filter(|f| f.is_finite()).unwrap_or(dflt)
}

fn opt_usize(v: &Json, key: &str, dflt: usize) -> usize {
    v.get(key).and_then(|x| x.as_usize()).unwrap_or(dflt)
}

fn ledger_json(l: &Ledger, p: &Prices, wall_s: f64) -> Json {
    let (fs, fr, fw) = l.shares(p);
    Json::obj(vec![
        ("node_updates", Json::n(l.samples as f64)),
        ("reads", Json::n(l.reads as f64)),
        ("writes", Json::n(l.writes as f64)),
        ("joules_z1_spice", Json::n(l.joules(p))),
        (
            "share",
            Json::obj(vec![
                ("sample", Json::n(fs)),
                ("read", Json::n(fr)),
                ("write", Json::n(fw)),
            ]),
        ),
        ("wall_seconds", Json::n(wall_s)),
        (
            "note",
            Json::s(
                "joules are Z1-class SPICE prices from arXiv:2608.01615 Table IV, applied to \
                 counted operations. They price the modelled device, not this CPU.",
            ),
        ),
    ])
}

fn certificate_json(c: &Certificate, capped: Option<usize>) -> Json {
    let mut out = vec![
        ("draws", Json::n(c.draws as f64)),
        ("beta_requested", Json::n(c.beta_requested)),
        ("beta_effective", Json::n(c.beta_eff)),
        (
            "beta_ci95",
            Json::Arr(vec![Json::n(c.beta_ci.0), Json::n(c.beta_ci.1)]),
        ),
        ("autocorrelation_time", Json::n(c.tau_int)),
        ("effective_sample_size", Json::n(c.ess)),
        ("passed", Json::Bool(c.passed())),
        (
            "findings",
            Json::Arr(c.findings.iter().map(|f| Json::Str(f.to_string())).collect()),
        ),
    ];
    if let (Some(tv), Some(fl)) = (c.tv_exact, c.noise_floor) {
        out.push(("total_variation", Json::n(tv)));
        out.push(("noise_floor", Json::n(fl)));
    }
    if let Some(n) = capped {
        out.push((
            "draws_capped_to",
            Json::n(n as f64),
        ));
        out.push((
            "cap_note",
            Json::s(
                "retained draws were reduced to bound memory on a large graph; a thinner \
                 certificate is reported rather than a fabricated one",
            ),
        ));
    }
    out.push((
        "note",
        Json::s(
            "computed from the returned samples alone, not from the sampler's own account of \
             itself. An empty findings list is the only thing that means passed.",
        ),
    ));
    Json::Obj(out.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn state_json(s: &[i8]) -> Json {
    Json::Arr(s.iter().map(|&x| Json::n(x as f64)).collect())
}

// ---- handlers ---------------------------------------------------------------------------------

/// Draw samples by chromatic block-Gibbs at fixed inverse temperature.
pub fn sample(req: &Json) -> Result<Json, String> {
    let gv = req.get("graph").ok_or("missing \"graph\"")?;
    let g = graph_from(gv)?;
    let beta = opt_f64(req, "beta", 1.0);
    if beta < 0.0 {
        return Err("\"beta\" must be non-negative (beta = 1/T)".into());
    }
    let sweeps = opt_usize(req, "sweeps", 100);
    let seed = req.get("seed").and_then(|s| s.as_u64()).unwrap_or(0);
    let threads = opt_usize(req, "threads", 1).max(1);

    let budget = (g.n as u64).saturating_mul(sweeps as u64);
    if budget > MAX_NODE_UPDATES {
        return Err(format!(
            "{} nodes x {sweeps} sweeps = {budget} node updates, over the {MAX_NODE_UPDATES} \
             ceiling; lower \"sweeps\" or split the run",
            g.n
        ));
    }

    let t0 = Instant::now();
    let mut led = Ledger::default();
    let mut smp = Sampler::new(&g, beta, seed);
    if let Some(cl) = req.get("clamp").and_then(|c| c.as_arr()) {
        for (k, e) in cl.iter().enumerate() {
            let t = e.as_arr().ok_or_else(|| format!("clamp {k} must be [i, value]"))?;
            if t.len() != 2 {
                return Err(format!("clamp {k} must be [i, value]"));
            }
            let i = idx(&t[0], g.n, k, "i")?;
            let val = t[1].as_f64().ok_or_else(|| format!("clamp {k}: value must be -1 or +1"))?;
            if val != 1.0 && val != -1.0 {
                return Err(format!("clamp {k}: value must be -1 or +1, got {val}"));
            }
            smp.clamp(i, val as i8);
        }
    }
    // `sweeps` is burn-in; then draws are recorded so the run can be certified. A sampler that
    // returns one state cannot be checked at all, which is why this is not optional.
    if threads > 1 {
        smp.sweeps_par(sweeps, threads, Some(&mut led));
    } else {
        smp.sweeps(sweeps, Some(&mut led));
    }

    let want_draws = opt_usize(req, "draws", 128).max(1);
    let thin = opt_usize(req, "thin", 1).max(1);
    // Retaining d draws of an n-spin graph costs d*n bytes. Rather than refuse to certify a large
    // model or quietly allocate a gigabyte, the draw count is reduced and the reduction is reported.
    let max_draws = (RETAINED_SPIN_BUDGET / g.n.max(1)).max(16);
    let draws = want_draws.min(max_draws);
    let capped = if draws < want_draws { Some(draws) } else { None };

    let mut samples: Vec<Vec<i8>> = Vec::with_capacity(draws);
    let mut trace: Vec<f64> = Vec::with_capacity(draws);
    for _ in 0..draws {
        if threads > 1 {
            smp.sweeps_par(thin, threads, Some(&mut led));
        } else {
            smp.sweeps(thin, Some(&mut led));
        }
        samples.push(smp.s.clone());
        trace.push(g.energy(&smp.s));
    }
    let cert = certify(&g, beta, &samples, &trace);

    let s = smp.read_all(Some(&mut led));
    let wall = t0.elapsed().as_secs_f64();

    let e = g.energy(&s);
    let m = s.iter().map(|&x| x as f64).sum::<f64>() / g.n as f64;
    let mut out = vec![
        ("nodes", Json::n(g.n as f64)),
        ("beta", Json::n(beta)),
        ("sweeps", Json::n(sweeps as f64)),
        ("seed", Json::n(seed as f64)),
        ("threads", Json::n(threads as f64)),
        ("energy", Json::n(e)),
        ("magnetization", Json::n(m)),
        ("ledger", ledger_json(&led, &Z1_SPICE, wall)),
        ("certificate", certificate_json(&cert, capped)),
    ];
    // A million-node state is not something to paste into a chat transcript; it is returned only
    // when asked for, and the summary statistics above are what a caller usually wants.
    if req.get("return_state").and_then(|b| b.as_bool()).unwrap_or(g.n <= 4096) {
        out.push(("state", state_json(&s)));
    } else {
        out.push(("state_omitted", Json::s("pass \"return_state\": true to include it")));
    }
    Ok(Json::Obj(out.into_iter().map(|(k, v)| (k.to_string(), v)).collect()))
}

/// Simulated annealing down a geometric beta ladder, tracking the best state seen.
pub fn anneal(req: &Json) -> Result<Json, String> {
    let gv = req.get("graph").ok_or("missing \"graph\"")?;
    let g = graph_from(gv)?;
    let beta_min = opt_f64(req, "beta_min", 0.1);
    let beta_max = opt_f64(req, "beta_max", 3.0);
    let stages = opt_usize(req, "stages", 24).max(2);
    let per = opt_usize(req, "sweeps_per_stage", 20).max(1);
    let seed = req.get("seed").and_then(|s| s.as_u64()).unwrap_or(0);
    if !(beta_min > 0.0 && beta_max > beta_min) {
        return Err("need 0 < beta_min < beta_max".into());
    }
    let budget = (g.n as u64).saturating_mul((stages * per) as u64);
    if budget > MAX_NODE_UPDATES {
        return Err(format!(
            "{budget} node updates requested, over the {MAX_NODE_UPDATES} ceiling"
        ));
    }

    let t0 = Instant::now();
    let ladder = ferrotherm::tempering::geometric_ladder(beta_min, beta_max, stages);
    let schedule: Vec<(f64, usize)> = ladder.iter().map(|&b| (b, per)).collect();
    let mut led = Ledger::default();
    let (best, best_e) = ferrotherm::tempering::anneal(&g, &schedule, seed, Some(&mut led));
    let wall = t0.elapsed().as_secs_f64();

    let mut out = vec![
        ("nodes", Json::n(g.n as f64)),
        ("best_energy", Json::n(best_e)),
        ("beta_min", Json::n(beta_min)),
        ("beta_max", Json::n(beta_max)),
        ("stages", Json::n(stages as f64)),
        ("sweeps_per_stage", Json::n(per as f64)),
        ("seed", Json::n(seed as f64)),
        ("ledger", ledger_json(&led, &Z1_SPICE, wall)),
    ];
    if req.get("return_state").and_then(|b| b.as_bool()).unwrap_or(g.n <= 4096) {
        out.push(("state", state_json(&best)));
    } else {
        out.push(("state_omitted", Json::s("pass \"return_state\": true to include it")));
    }
    Ok(Json::Obj(out.into_iter().map(|(k, v)| (k.to_string(), v)).collect()))
}

/// Energy of a supplied state under a supplied graph.
pub fn energy(req: &Json) -> Result<Json, String> {
    let g = graph_from(req.get("graph").ok_or("missing \"graph\"")?)?;
    let sv = req.get("state").and_then(|s| s.as_arr()).ok_or("missing \"state\" array")?;
    if sv.len() != g.n {
        return Err(format!("state has {} entries but the graph has {} nodes", sv.len(), g.n));
    }
    let mut s = Vec::with_capacity(g.n);
    for (i, x) in sv.iter().enumerate() {
        let v = x.as_f64().ok_or_else(|| format!("state[{i}] must be -1 or +1"))?;
        if v != 1.0 && v != -1.0 {
            return Err(format!("state[{i}] must be -1 or +1, got {v}"));
        }
        s.push(v as i8);
    }
    Ok(Json::obj(vec![
        ("energy", Json::n(g.energy(&s))),
        (
            "magnetization",
            Json::n(s.iter().map(|&x| x as f64).sum::<f64>() / g.n as f64),
        ),
    ]))
}

/// Enumerate the exact Boltzmann distribution and compare the sampler against it.
///
/// This is the tool that lets a caller check the sampler rather than trust it. It is capped at 20
/// nodes because the enumeration is 2^n.
pub fn verify(req: &Json) -> Result<Json, String> {
    let g = graph_from(req.get("graph").ok_or("missing \"graph\"")?)?;
    if g.n > 20 {
        return Err(format!(
            "exact verification enumerates 2^n states and is capped at 20 nodes; this graph has {}",
            g.n
        ));
    }
    let beta = opt_f64(req, "beta", 1.0);
    let sweeps = opt_usize(req, "sweeps", 200);
    let draws = opt_usize(req, "draws", 20_000);
    // Sweeps between recorded draws. Consecutive draws from one chain are correlated, and at high
    // beta that correlation biases the histogram badly enough to fail verification on a sampler
    // that is in fact correct. Thinning is the cure; the autocorrelation is the disease.
    let thin = opt_usize(req, "thin", 1).max(1);
    let seed = req.get("seed").and_then(|s| s.as_u64()).unwrap_or(0);

    let t0 = Instant::now();
    let exact = ising::exact_boltzmann(&g, beta);
    let mut hist = vec![0u64; 1 << g.n];
    let mut led = Ledger::default();
    let mut smp = Sampler::new(&g, beta, seed);
    smp.sweeps(sweeps, Some(&mut led)); // burn in
    for _ in 0..draws {
        smp.sweeps(thin, Some(&mut led));
        let s = smp.read_all(Some(&mut led));
        let mut k = 0usize;
        for (b, &v) in s.iter().enumerate() {
            if v > 0 {
                k |= 1 << b;
            }
        }
        hist[k] += 1;
    }
    let emp: Vec<f64> = hist.iter().map(|&c| c as f64 / draws as f64).collect();
    let tv = ising::tv(&emp, &exact);
    let wall = t0.elapsed().as_secs_f64();

    Ok(Json::obj(vec![
        ("nodes", Json::n(g.n as f64)),
        ("beta", Json::n(beta)),
        ("draws", Json::n(draws as f64)),
        ("thin", Json::n(thin as f64)),
        ("total_variation_distance", Json::n(tv)),
        (
            "expected_sampling_noise",
            Json::n(((1u64 << g.n) as f64 / draws as f64).sqrt() * 0.5),
        ),
        ("ledger", ledger_json(&led, &Z1_SPICE, wall)),
        (
            "note",
            Json::s(
                "TV distance between the sampler's empirical distribution and the enumerated \
                 Boltzmann distribution. Compare it against expected_sampling_noise: a TV below \
                 that floor is agreement, not accuracy you can quote. A TV above the floor at high \
                 beta usually means correlated draws rather than a wrong sampler: raise \"thin\".",
            ),
        ),
    ]))
}

/// Everything a caller needs to use this server without reading its source.
pub fn capabilities() -> Json {
    Json::obj(vec![
        ("name", Json::s("ferrotherm")),
        ("version", Json::s(env!("CARGO_PKG_VERSION"))),
        (
            "description",
            Json::s(
                "Thermodynamic sampling: draw from Boltzmann distributions over Ising graphs by \
                 chromatic block-Gibbs, anneal them for optimisation, and price every run in \
                 device joules.",
            ),
        ),
        (
            "unit_terminology",
            Json::s(
                "The sampled unit is a binary stochastic neuron (machine learning) or an Ising \
                 spin under Glauber dynamics (statistical physics). The 2016 coinage \"p-bit\" \
                 names the same object.",
            ),
        ),
        (
            "operations",
            Json::Arr(vec![
                op("sample", "Draw a state by block-Gibbs at fixed beta.", "graph, beta, sweeps, seed, threads, clamp, return_state"),
                op("anneal", "Minimise energy down a geometric beta ladder.", "graph, beta_min, beta_max, stages, sweeps_per_stage, seed"),
                op("energy", "Energy and magnetization of a given state.", "graph, state"),
                op("verify", "Compare the sampler to the exact distribution (n <= 20).", "graph, beta, draws, sweeps, seed"),
            ]),
        ),
        (
            "graph_spec",
            Json::obj(vec![
                ("explicit", Json::s("{\"n\": 4, \"couplings\": [[0,1,1.0]], \"biases\": [[0,0.5]]}")),
                ("builtin_ring", Json::s("{\"builtin\": \"ring\", \"n\": 16, \"j\": 1.0, \"h\": 0.0}")),
                ("builtin_lattice", Json::s("{\"builtin\": \"lattice2d\", \"l\": 32, \"j\": 1.0}")),
                ("convention", Json::s("States are -1/+1. Energy is -sum_ij J_ij s_i s_j - sum_i h_i s_i.")),
            ]),
        ),
        (
            "limits",
            Json::obj(vec![
                ("max_nodes", Json::n(MAX_NODES as f64)),
                ("max_node_updates_per_request", Json::n(MAX_NODE_UPDATES as f64)),
                ("exact_verification_max_nodes", Json::n(20.0)),
            ]),
        ),
        (
            "determinism",
            Json::s("Same seed and same thread count reproduce a run bit for bit."),
        ),
    ])
}

fn op(name: &str, what: &str, fields: &str) -> Json {
    Json::obj(vec![
        ("name", Json::s(name)),
        ("description", Json::s(what)),
        ("fields", Json::s(fields)),
    ])
}

/// Route a named operation. Shared by both transports.
pub fn dispatch(op: &str, req: &Json) -> Result<Json, String> {
    match op {
        "sample" => sample(req),
        "anneal" => anneal(req),
        "energy" => energy(req),
        "verify" => verify(req),
        "capabilities" => Ok(capabilities()),
        other => Err(format!(
            "unknown operation {other:?}; call \"capabilities\" for the list"
        )),
    }
}

/// Parse a request body, with the error phrased for whoever has to fix it.
pub fn parse_body(body: &str) -> Result<Json, String> {
    if body.trim().is_empty() {
        return Ok(Json::Obj(Vec::new()));
    }
    parse(body).map_err(|e| format!("request body is not valid JSON: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::parse;

    fn run(op: &str, body: &str) -> Result<Json, String> {
        dispatch(op, &parse(body).unwrap())
    }

    #[test]
    fn samples_a_builtin_lattice() {
        let r = run("sample", r#"{"graph":{"builtin":"lattice2d","l":8},"beta":0.5,"sweeps":50}"#)
            .unwrap();
        assert_eq!(r.get("nodes").unwrap().as_f64(), Some(64.0));
        assert_eq!(r.get("state").unwrap().as_arr().unwrap().len(), 64);
        // Certification is not free and the ledger says so: 50 burn-in sweeps plus 128 recorded
        // draws at thin 1. A run that returns one state cannot be checked, so this is the price.
        let led = r.get("ledger").unwrap();
        assert_eq!(led.get("node_updates").unwrap().as_f64(), Some(64.0 * (50.0 + 128.0)));

        let c = r.get("certificate").expect("every sample must carry a certificate");
        assert_eq!(c.get("draws").unwrap().as_f64(), Some(128.0));
        assert!(c.get("beta_effective").is_some());
        assert!(c.get("passed").unwrap().as_bool().is_some());
    }

    #[test]
    fn same_seed_reproduces() {
        let b = r#"{"graph":{"builtin":"ring","n":24},"beta":0.8,"sweeps":30,"seed":7}"#;
        let (x, y) = (run("sample", b).unwrap(), run("sample", b).unwrap());
        // everything but the clock: wall_seconds is a measurement of this machine, not of the run
        for k in ["state", "energy", "magnetization"] {
            assert_eq!(x.get(k), y.get(k), "{k} differs between identical runs");
        }
        for k in ["node_updates", "reads", "writes", "joules_z1_spice"] {
            assert_eq!(x.get("ledger").unwrap().get(k), y.get("ledger").unwrap().get(k), "ledger {k}");
        }
    }

    #[test]
    fn sampler_matches_the_exact_distribution() {
        // The whole point of the verify tool: it must pass on a graph we can enumerate. The bar is
        // the sampling noise floor the tool itself reports, not a number picked by hand -- with
        // 40k draws over 256 states, finite-sample scatter alone puts TV near 0.04.
        let r = run("verify", r#"{"graph":{"builtin":"ring","n":8,"j":1.0},"beta":0.4,"draws":40000}"#)
            .unwrap();
        let tv = r.get("total_variation_distance").unwrap().as_f64().unwrap();
        let floor = r.get("expected_sampling_noise").unwrap().as_f64().unwrap();
        assert!(tv < floor, "TV {tv} exceeds the {floor} noise floor; sampler disagrees with exact");
    }

    #[test]
    fn verification_detects_correlated_draws_and_thinning_fixes_them() {
        // A check that only ever passes proves nothing. At beta 1.6 the ring is strongly ordered
        // and single-spin Glauber decorrelates slowly, so back-to-back draws are correlated and
        // the histogram is biased above the noise floor even though the sampler is correct. That
        // is a real effect worth catching, and thinning the chain is the cure -- which makes this
        // both the negative control for the check above and the demonstration of "thin".
        let g = r#"{"builtin":"ring","n":8,"j":1.0}"#;
        let tight = run("verify", &format!(r#"{{"graph":{g},"beta":1.6,"draws":40000,"thin":1}}"#))
            .unwrap();
        let thinned = run("verify", &format!(r#"{{"graph":{g},"beta":1.6,"draws":40000,"thin":40}}"#))
            .unwrap();
        let floor = tight.get("expected_sampling_noise").unwrap().as_f64().unwrap();
        let (a, b) = (
            tight.get("total_variation_distance").unwrap().as_f64().unwrap(),
            thinned.get("total_variation_distance").unwrap().as_f64().unwrap(),
        );
        assert!(a > floor, "correlated draws should exceed the {floor} floor, got {a}");
        assert!(b < floor, "thinning to 40 sweeps should drop under the {floor} floor, got {b}");
    }

    #[test]
    fn energy_agrees_with_the_sampler() {
        let s = run("sample", r#"{"graph":{"builtin":"ring","n":12},"beta":1.0,"seed":3}"#).unwrap();
        let state = crate::json::write(s.get("state").unwrap());
        let e = run("energy", &format!(r#"{{"graph":{{"builtin":"ring","n":12}},"state":{state}}}"#))
            .unwrap();
        assert_eq!(e.get("energy").unwrap().as_f64(), s.get("energy").unwrap().as_f64());
    }

    #[test]
    fn anneal_beats_a_hot_sample() {
        let g = r#"{"builtin":"lattice2d","l":10,"j":1.0}"#;
        let a = run("anneal", &format!(r#"{{"graph":{g},"beta_min":0.05,"beta_max":4.0,"stages":30}}"#))
            .unwrap();
        let hot = run("sample", &format!(r#"{{"graph":{g},"beta":0.05,"sweeps":600}}"#)).unwrap();
        let (ea, eh) = (
            a.get("best_energy").unwrap().as_f64().unwrap(),
            hot.get("energy").unwrap().as_f64().unwrap(),
        );
        assert!(ea < eh, "annealed {ea} should beat hot sample {eh}");
    }

    #[test]
    fn errors_say_how_to_fix_them() {
        let e = run("sample", r#"{"graph":{"n":4,"couplings":[[0,9,1.0]]}}"#).unwrap_err();
        assert!(e.contains("out of range"), "{e}");
        let e = run("sample", r#"{"graph":{"builtin":"nope","n":4}}"#).unwrap_err();
        assert!(e.contains("ring") && e.contains("lattice2d"), "{e}");
        let e = run("sample", r#"{}"#).unwrap_err();
        assert!(e.contains("graph"), "{e}");
        let e = run("sample", r#"{"graph":{"n":4,"couplings":[[0,0,1.0]]}}"#).unwrap_err();
        assert!(e.contains("itself"), "{e}");
    }

    #[test]
    fn refuses_a_job_it_cannot_finish() {
        let e = run("sample", r#"{"graph":{"builtin":"lattice2d","l":2000},"sweeps":100000}"#)
            .unwrap_err();
        assert!(e.contains("ceiling"), "{e}");
    }

    #[test]
    fn clamped_nodes_hold_their_value() {
        let r = run(
            "sample",
            r#"{"graph":{"builtin":"ring","n":16},"beta":1.0,"sweeps":50,"clamp":[[0,1],[8,-1]]}"#,
        )
        .unwrap();
        let s = r.get("state").unwrap().as_arr().unwrap();
        assert_eq!(s[0].as_f64(), Some(1.0));
        assert_eq!(s[8].as_f64(), Some(-1.0));
    }

    #[test]
    fn a_certificate_can_fail_through_the_api() {
        // The whole point. A cold lattice sampled with no thinning must come back flagged, not
        // silently blessed, or the field is decoration.
        let r = run(
            "sample",
            r#"{"graph":{"builtin":"lattice2d","l":24},"beta":0.7,"sweeps":0,"draws":400,"thin":1,
                "return_state":false}"#,
        )
        .unwrap();
        let c = r.get("certificate").unwrap();
        assert_eq!(c.get("passed").unwrap().as_bool(), Some(false));
        assert!(!c.get("findings").unwrap().as_arr().unwrap().is_empty());
    }

    #[test]
    fn a_large_graph_caps_its_draws_and_says_so() {
        // Rather than allocate a gigabyte or refuse to certify, it certifies thinner and reports it.
        let r = run(
            "sample",
            r#"{"graph":{"builtin":"lattice2d","l":600},"sweeps":1,"draws":1000}"#,
        )
        .unwrap();
        let c = r.get("certificate").unwrap();
        assert!(c.get("draws_capped_to").is_some(), "a 360k-spin graph must cap its draws");
        assert!(c.get("draws").unwrap().as_f64().unwrap() < 1000.0);
    }

    #[test]
    fn large_graphs_omit_the_state_by_default() {
        let r = run("sample", r#"{"graph":{"builtin":"lattice2d","l":100},"sweeps":2}"#).unwrap();
        assert!(r.get("state").is_none());
        assert!(r.get("state_omitted").is_some());
    }
}
