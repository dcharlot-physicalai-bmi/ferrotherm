//! The operations, once, so that HTTP and MCP cannot drift apart.
//!
//! Every handler takes a JSON request and returns a JSON result or an error string. The transports
//! in `http` and `mcp` do nothing but move bytes and shape envelopes.

use crate::json::{parse, Json};
use ferrotherm::gibbs::Sampler;
use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::ising;
use ferrotherm::certify::{certify, Certificate};
use ferrotherm::model::{Constraint, Expr, Lit, Model, Sense};
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

/// Refuse a run whose TOTAL cost exceeds the ceiling.
///
/// The whole cost, which is the part that was wrong: the check counted burn-in and stopped there,
/// while every one of these handlers then runs `draws x thin` further sweeps to have something to
/// certify. `{"sweeps": 1, "draws": 128, "thin": 2000}` declared 1,024 node updates at the gate and
/// did a quarter of a billion. A ceiling that only looks at the cheap half is not a ceiling.
fn bound_updates(n: usize, sweeps: usize, draws: usize, thin: usize) -> Result<(), String> {
    let recorded = (draws as u64).saturating_mul(thin as u64);
    let total = (n as u64).saturating_mul((sweeps as u64).saturating_add(recorded));
    if total > MAX_NODE_UPDATES {
        return Err(format!(
            "{n} nodes x ({sweeps} burn-in + {draws} draws x {thin} thin) = {total} node updates, \
             over the {MAX_NODE_UPDATES} ceiling; lower \"sweeps\", \"draws\" or \"thin\", or \
             split the run"
        ));
    }
    Ok(())
}

/// A modeller's value: absent means `dflt`, present-but-unreadable is an error.
///
/// `.and_then(as_i64).unwrap_or(dflt)` reads the two cases as one, so `"value": "13"` -- a JSON
/// string where a number belongs, which is what a templating layer or a shell pipeline produces --
/// became `dflt` and the caller got a confident answer to a question they did not ask.
fn value_of(v: &Json, key: &str, dflt: i64, what: &str) -> Result<i64, String> {
    match v.get(key) {
        None => Ok(dflt),
        Some(x) => x.as_i64().ok_or_else(|| {
            format!("{what}: \"{key}\" must be a whole number, not {}", describe(x))
        }),
    }
}

/// What a JSON value is, for an error message that tells the caller what they actually sent.
fn describe(v: &Json) -> &'static str {
    match v {
        Json::Null => "null",
        Json::Bool(_) => "a boolean",
        Json::Num(_) => "a fractional number",
        Json::Str(_) => "a string",
        Json::Arr(_) => "an array",
        Json::Obj(_) => "an object",
    }
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

    // Read the recording parameters BEFORE the ceiling, because they are most of the cost.
    let want_draws = opt_usize(req, "draws", 128).max(1);
    let thin = opt_usize(req, "thin", 1).max(1);
    // Retaining d draws of an n-spin graph costs d*n bytes. Rather than refuse to certify a large
    // model or quietly allocate a gigabyte, the draw count is reduced and the reduction is reported.
    let max_draws = (RETAINED_SPIN_BUDGET / g.n.max(1)).max(16);
    let draws = want_draws.min(max_draws);
    let capped = if draws < want_draws { Some(draws) } else { None };
    bound_updates(g.n, sweeps, draws, thin)?;

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
    // Capped at 20 nodes, so the graph is small -- but `draws` defaults to 20,000 and `thin` is a
    // caller's number, so the sweep count is not bounded by the node count.
    bound_updates(g.n, sweeps, draws, thin)?;

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
                op("solve", "State a problem with named variables and constraints; get named values back.", "variables, constraints, objective, tries, penalty, schedule"),
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

/// Solve a model stated in the problem's own vocabulary.
///
/// This is the operation an agent should reach for. Every other endpoint takes a graph of spins and
/// gives back an array of ±1; this one takes named variables with domains and constraints, and
/// gives back **named values**. An agent that has to compute spin indices to express "these two
/// must differ" is doing the compiler's job.
pub fn solve(req: &Json) -> Result<Json, String> {
    let vars = req
        .get("variables")
        .and_then(|v| v.as_arr())
        .ok_or("missing \"variables\": an array of {name, values} or {name, lo, hi}")?;
    if vars.is_empty() {
        return Err("a model needs at least one variable".into());
    }

    let mut m = Model::new();
    let mut handles = Vec::new();
    let mut names = Vec::new();
    for (i, v) in vars.iter().enumerate() {
        let name = v
            .get("name")
            .and_then(|n| n.as_str())
            .ok_or_else(|| format!("variable {i} needs a \"name\""))?
            .to_string();
        if names.contains(&name) {
            return Err(format!("two variables are both named {name:?}"));
        }
        let h = match (v.get("values"), v.get("lo"), v.get("hi")) {
            (Some(k), _, _) => {
                let k = k.as_usize().ok_or_else(|| format!("{name}: \"values\" must be an integer"))?;
                if k < 2 {
                    return Err(format!("{name}: a variable with {k} values is a constant"));
                }
                m.categorical(&name, k)
            }
            (None, Some(lo), Some(hi)) => {
                let (lo, hi) = (
                    lo.as_f64().ok_or("\"lo\" must be a number")? as i64,
                    hi.as_f64().ok_or("\"hi\" must be a number")? as i64,
                );
                if hi <= lo {
                    return Err(format!("{name}: an integer range needs hi > lo"));
                }
                m.integer(&name, lo, hi)
            }
            _ => return Err(format!("{name}: give either \"values\" or both \"lo\" and \"hi\"")),
        };
        handles.push(h);
        names.push(name);
    }

    let find = |n: &str| -> Result<usize, String> {
        names.iter().position(|x| x == n).ok_or_else(|| {
            format!("no variable named {n:?}; declared: {}", names.join(", "))
        })
    };

    if let Some(cs) = req.get("constraints").and_then(|c| c.as_arr()) {
        for (i, c) in cs.iter().enumerate() {
            let kind = c
                .get("type")
                .and_then(|k| k.as_str())
                .ok_or_else(|| format!("constraint {i} needs a \"type\""))?;
            match kind {
                "not_equal" | "equal" => {
                    let a = find(c.get("a").and_then(|x| x.as_str()).unwrap_or(""))?;
                    let b = find(c.get("b").and_then(|x| x.as_str()).unwrap_or(""))?;
                    if a == b {
                        // The C ABI refuses this; the JSON surface used to accept it and return a
                        // confident feasible answer to an unsatisfiable request.
                        return Err(format!(
                            "{kind}: a variable cannot be compared with itself (both sides name \"{}\")",
                            c.get("a").and_then(|x| x.as_str()).unwrap_or("")
                        ));
                    }
                    if kind == "not_equal" {
                        m.not_equal(handles[a], handles[b]);
                    } else {
                        m.equal(handles[a], handles[b]);
                    }
                }
                "fix" => {
                    let v = find(c.get("var").and_then(|x| x.as_str()).unwrap_or(""))?;
                    let val = value_of(c, "value", 0, "fix")?;
                    m.fix(handles[v], val);
                }
                "at_most" | "at_least" => {
                    let k = c.get("k").and_then(|x| x.as_usize())
                        .ok_or_else(|| format!("{kind} needs \"k\""))?;
                    let items = c.get("of").and_then(|x| x.as_arr())
                        .ok_or_else(|| format!("{kind} needs \"of\""))?;
                    let mut lits = Vec::new();
                    for it in items {
                        let vn = it.get("var").and_then(|x| x.as_str()).unwrap_or("");
                        let vv = value_of(it, "value", 1, kind)?;
                        lits.push(Lit::Is(handles[find(vn)?], vv));
                    }
                    if kind == "at_most" {
                        m.at_most(lits, k);
                    } else {
                        m.at_least(lits, k);
                    }
                }
                "exactly_one" | "at_most_one" => {
                    let items = c.get("of").and_then(|x| x.as_arr())
                        .ok_or_else(|| format!("{kind} needs \"of\""))?;
                    let mut lits = Vec::new();
                    for it in items {
                        let vn = it.get("var").and_then(|x| x.as_str()).unwrap_or("");
                        let vv = value_of(it, "value", 1, kind)?;
                        lits.push(Lit::Is(handles[find(vn)?], vv));
                    }
                    if lits.len() < 2 {
                        return Err(format!("{kind} needs at least two things to count"));
                    }
                    // Pairwise exclusion rather than a squared count: no slack variable, and it is
                    // the cheaper lowering whenever k is one.
                    m.constrain(if kind == "exactly_one" {
                        Constraint::ExactlyOne(lits)
                    } else {
                        Constraint::AtMostOne(lits)
                    });
                }
                "cardinality" => {
                    let k = c.get("k").and_then(|x| x.as_usize()).ok_or("cardinality needs \"k\"")?;
                    let items = c.get("of").and_then(|x| x.as_arr()).ok_or("cardinality needs \"of\"")?;
                    let mut lits = Vec::new();
                    for it in items {
                        let vn = it.get("var").and_then(|x| x.as_str()).unwrap_or("");
                        let vv = value_of(it, "value", 1, "cardinality")?;
                        lits.push(Lit::Is(handles[find(vn)?], vv));
                    }
                    m.cardinality(lits, k);
                }
                other => {
                    return Err(format!(
                        "unknown constraint {other:?}; known: not_equal, equal, fix, \
                         cardinality, at_most, at_least, exactly_one, at_most_one"
                    ))
                }
            }
        }
    }

    if let Some(o) = req.get("objective") {
        // Not `.unwrap_or(false)`. A "maximize" the reader cannot understand used to become
        // MINIMIZE, which is not a degraded answer -- it is the opposite one, returned with
        // feasible: true and nothing to suggest anything went wrong.
        let maximize = match o.get("maximize") {
            None => false,
            Some(x) => x.as_bool().ok_or_else(|| {
                format!("objective: \"maximize\" must be true or false, not {}", describe(x))
            })?,
        };
        let terms = o.get("terms").and_then(|x| x.as_arr()).ok_or("objective needs \"terms\"")?;
        let mut e = Expr::zero();
        for t in terms {
            let w = t.get("weight").and_then(|x| x.as_f64()).unwrap_or(1.0);

            // A term is one variable, a pair, or a PRODUCT of any number written as "of". The pair
            // form stays: it reads well for the common case, and removing it would break callers.
            if let Some(items) = t.get("of").and_then(|x| x.as_arr()) {
                if items.is_empty() {
                    return Err("an objective term's \"of\" must not be empty".into());
                }
                let mut lits = Vec::new();
                for it in items {
                    let n = it.get("var").and_then(|x| x.as_str()).unwrap_or("");
                    let v = value_of(it, "value", 1, "objective term")?;
                    lits.push(Lit::Is(handles[find(n)?], v));
                }
                e = e.plus(Expr::product(w, &lits));
                continue;
            }

            let vn = t.get("var").and_then(|x| x.as_str()).unwrap_or("");
            let vv = value_of(t, "value", 1, "objective term")?;
            let a = Lit::Is(handles[find(vn)?], vv);
            e = match t.get("and_var").and_then(|x| x.as_str()) {
                Some(bn) => {
                    let bv = value_of(t, "and_value", vv, "objective term")?;
                    e.plus(Expr::pair(w, a, Lit::Is(handles[find(bn)?], bv)))
                }
                None => e.plus(Expr::lit(w, a)),
            };
        }
        m.objective(if maximize { Sense::Maximize } else { Sense::Minimize }, e);
    }

    if let Some(p) = req.get("penalty").and_then(|x| x.as_f64()) {
        m.fixed_penalty(p);
    }

    let penalty = m.effective_penalty();
    let compiled = m.compile().map_err(|e| e.to_string())?;
    let tries = opt_usize(req, "tries", 16).clamp(1, 500);

    // The annealing ladder, which the default handles for the models people write first and not for
    // the largest they will write. Every other surface lets a caller who measured their instance say
    // so; this one did not, and the only advice on an infeasible answer was to raise the penalty --
    // which does nothing for a model that is simply not being annealed long enough.
    let d = ferrotherm::model::Compiled::DEFAULT_LADDER;
    let ladder = req.get("schedule");
    let sched = match ladder {
        None => None,
        Some(s) => {
            let hot = s.get("beta_hot").and_then(|x| x.as_f64()).unwrap_or(d.0);
            let cold = s.get("beta_cold").and_then(|x| x.as_f64()).unwrap_or(d.1);
            let stages = opt_usize(s, "stages", d.2).clamp(2, 10_000);
            let per = opt_usize(s, "sweeps", d.3).clamp(1, 100_000);
            if !hot.is_finite() || !cold.is_finite() || cold <= hot {
                return Err(format!(
                    "schedule: \"beta_cold\" ({cold}) must exceed \"beta_hot\" ({hot}), and both \
                     must be real numbers. Annealing runs hot to cold."
                ));
            }
            bound_updates(compiled.spins(), stages * per * tries, 0, 1)?;
            Some(ferrotherm::schedule::Schedule::geometric(hot, cold, stages, per))
        }
    };

    let t0 = Instant::now();
    let sol = match &sched {
        Some(s) => compiled.solve_best_with(s, tries as u64),
        None => compiled.solve_best_of(tries as u64),
    };
    let wall = t0.elapsed().as_secs_f64();

    let mut values = Vec::new();
    for n in &names {
        if let Some(v) = sol.get(n) {
            values.push((n.clone(), Json::n(v as f64)));
        }
    }

    Ok(Json::obj(vec![
        ("values", Json::Obj(values)),
        ("feasible", Json::Bool(sol.feasible())),
        (
            "did_not_decode",
            Json::Arr(sol.invalid.iter().map(|s| Json::s(s)).collect()),
        ),
        (
            // Every constraint the answer breaks, in the caller's own names. Distinct from
            // did_not_decode: a broken constraint means every value read cleanly and one of them
            // is not what was asked for, which is not visible from the values alone.
            "violated",
            // Each carries how far outside it sits, not only that it broke. A caller ranking
            // repairs, or deciding whether a larger penalty would be enough, needs the magnitude.
            Json::Arr(
                sol.violated
                    .iter()
                    .map(|v| {
                        Json::obj(vec![
                            ("constraint", Json::s(&v.detail)),
                            ("by", Json::n(v.amount)),
                        ])
                    })
                    .collect(),
            ),
        ),
        ("energy", Json::n(sol.energy)),
        ("spins", Json::n(compiled.spins() as f64)),
        // Spins the higher-order lowering added. Non-zero means the answer solves a model with more
        // spins than the variables required, and that the guarantee is about optima not sampling.
        ("ancillas", Json::n(compiled.ancillas as f64)),
        ("penalty", Json::n(penalty)),
        ("tries", Json::n(tries as f64)),
        ("wall_seconds", Json::n(wall)),
        ("ftp", Json::Str(compiled.program.to_ftp())),
        (
            "note",
            Json::s(
                "`feasible: false` means the answer breaks something you asked for: either a \
                 variable in \"did_not_decode\" whose encoding was violated, or a constraint in \
                 \"violated\" that the objective outbid. A penalty makes a constraint expensive, \
                 not impossible. Raise \"penalty\" or lower the objective weights and try again; \
                 if it stays infeasible at a large penalty, the problem itself may have no answer. \
                 The \"ftp\" field is the compiled program and runs unchanged on any backend.",
            ),
        ),
    ]))
}

/// Route a named operation. Shared by both transports.
pub fn dispatch(op: &str, req: &Json) -> Result<Json, String> {
    match op {
        "sample" => sample(req),
        "anneal" => anneal(req),
        "energy" => energy(req),
        "verify" => verify(req),
        "solve" => solve(req),
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

#[cfg(test)]
mod solve_tests {
    use super::*;
    use crate::json::parse;

    fn go(body: &str) -> Result<Json, String> {
        dispatch("solve", &parse(body).unwrap())
    }

    #[test]
    fn graph_colouring_in_the_problems_own_words() {
        let r = go(r#"{"variables":[{"name":"a","values":3},{"name":"b","values":3},
                       {"name":"c","values":3}],
                       "constraints":[{"type":"not_equal","a":"a","b":"b"},
                                      {"type":"not_equal","a":"b","b":"c"},
                                      {"type":"not_equal","a":"a","b":"c"}],"tries":20}"#)
            .unwrap();
        assert_eq!(r.get("feasible").unwrap().as_bool(), Some(true));
        let v = r.get("values").unwrap();
        let (a, b, c) = (
            v.get("a").unwrap().as_f64().unwrap(),
            v.get("b").unwrap().as_f64().unwrap(),
            v.get("c").unwrap().as_f64().unwrap(),
        );
        assert!(a != b && b != c && a != c, "a triangle needs three colours: {a} {b} {c}");
        assert!(r.get("ftp").unwrap().as_str().unwrap().starts_with("ftp 1"));
    }

    #[test]
    fn an_integer_comes_back_in_its_own_units() {
        let r = go(r#"{"variables":[{"name":"t","lo":10,"hi":20}],
                       "constraints":[{"type":"fix","var":"t","value":13}],"tries":8}"#)
            .unwrap();
        assert_eq!(r.get("values").unwrap().get("t").unwrap().as_f64(), Some(13.0));
    }

    #[test]
    fn a_value_outside_the_range_is_refused_by_name() {
        // A caller who writes a slot index where a value belongs gets told which is which, rather
        // than an answer to a question they did not ask.
        let e = go(r#"{"variables":[{"name":"t","lo":10,"hi":20}],
                       "constraints":[{"type":"fix","var":"t","value":3}],"tries":4}"#)
            .unwrap_err();
        assert!(e.contains("10..=20") && e.contains('t'), "{e}");
    }

    #[test]
    fn a_range_below_zero_survives_json() {
        // JSON numbers are signed and so are integer variables; the old reader could not carry a
        // negative value at all.
        let r = go(r#"{"variables":[{"name":"t","lo":-40,"hi":-10}],
                       "constraints":[{"type":"fix","var":"t","value":-25}],"tries":8}"#)
            .unwrap();
        assert_eq!(r.get("values").unwrap().get("t").unwrap().as_f64(), Some(-25.0));
    }

    #[test]
    fn cardinality_and_an_objective_together() {
        // Pick exactly two of five, preferring the most valuable.
        let r = go(r#"{"variables":[{"name":"b0","values":2},{"name":"b1","values":2},
                       {"name":"b2","values":2},{"name":"b3","values":2},{"name":"b4","values":2}],
                       "constraints":[{"type":"cardinality","k":2,"of":[
                         {"var":"b0","value":1},{"var":"b1","value":1},{"var":"b2","value":1},
                         {"var":"b3","value":1},{"var":"b4","value":1}]}],
                       "objective":{"maximize":true,"terms":[
                         {"var":"b3","value":1,"weight":3},{"var":"b4","value":1,"weight":4},
                         {"var":"b0","value":1,"weight":1}]},"tries":40}"#)
            .unwrap();
        assert_eq!(r.get("feasible").unwrap().as_bool(), Some(true));
        let v = r.get("values").unwrap();
        let on: Vec<&str> = ["b0", "b1", "b2", "b3", "b4"]
            .iter()
            .filter(|n| v.get(n).unwrap().as_f64() == Some(1.0))
            .copied()
            .collect();
        assert_eq!(on.len(), 2, "exactly two: {on:?}");
        assert!(on.contains(&"b3") && on.contains(&"b4"), "the valuable two: {on:?}");
    }

    #[test]
    fn the_penalty_scales_and_is_reported() {
        let r = go(r#"{"variables":[{"name":"x","values":4}],
                       "objective":{"maximize":true,"terms":[{"var":"x","value":3,"weight":9}]},
                       "tries":10}"#)
            .unwrap();
        assert_eq!(r.get("penalty").unwrap().as_f64(), Some(18.0), "twice the largest weight");
        assert_eq!(r.get("values").unwrap().get("x").unwrap().as_f64(), Some(3.0));
    }

    #[test]
    fn every_mistake_names_itself() {
        for (body, needle) in [
            (r#"{}"#, "variables"),
            (r#"{"variables":[]}"#, "at least one"),
            (r#"{"variables":[{"values":3}]}"#, "needs a \"name\""),
            (r#"{"variables":[{"name":"a","values":1}]}"#, "constant"),
            (r#"{"variables":[{"name":"a","lo":5,"hi":5}]}"#, "hi > lo"),
            (r#"{"variables":[{"name":"a","values":2},{"name":"a","values":2}]}"#, "both named"),
            (r#"{"variables":[{"name":"a","values":2}],"constraints":[{"type":"nope"}]}"#, "unknown constraint"),
            (r#"{"variables":[{"name":"a","values":2}],"constraints":[{"type":"equal","a":"a","b":"zz"}]}"#, "no variable named"),
        ] {
            let e = go(body).unwrap_err();
            assert!(e.contains(needle), "{body}\n  said: {e}\n  wanted: {needle}");
        }
    }
}

#[cfg(test)]
mod inequality_tests {
    use super::*;
    use crate::json::parse;

    fn go(body: &str) -> Result<Json, String> {
        dispatch("solve", &parse(body).unwrap())
    }

    #[test]
    fn at_most_caps_a_rewarded_selection() {
        // Everything is worth taking, so only the constraint stops it.
        let r = go(r#"{"variables":[{"name":"a","values":2},{"name":"b","values":2},
                       {"name":"c","values":2},{"name":"d","values":2}],
                       "constraints":[{"type":"at_most","k":2,"of":[
                         {"var":"a","value":1},{"var":"b","value":1},
                         {"var":"c","value":1},{"var":"d","value":1}]}],
                       "objective":{"maximize":true,"terms":[
                         {"var":"a","value":1},{"var":"b","value":1},
                         {"var":"c","value":1},{"var":"d","value":1}]},"tries":60}"#)
            .unwrap();
        assert_eq!(r.get("feasible").unwrap().as_bool(), Some(true));
        let v = r.get("values").unwrap();
        let on = ["a", "b", "c", "d"]
            .iter()
            .filter(|n| v.get(n).unwrap().as_f64() == Some(1.0))
            .count();
        assert_eq!(on, 2, "as many as allowed, no more");
    }

    #[test]
    fn at_least_forces_a_floor() {
        let r = go(r#"{"variables":[{"name":"a","values":2},{"name":"b","values":2},
                       {"name":"c","values":2},{"name":"d","values":2}],
                       "constraints":[{"type":"at_least","k":3,"of":[
                         {"var":"a","value":1},{"var":"b","value":1},
                         {"var":"c","value":1},{"var":"d","value":1}]}],
                       "objective":{"maximize":true,"terms":[
                         {"var":"a","value":0},{"var":"b","value":0},
                         {"var":"c","value":0},{"var":"d","value":0}]},"tries":60}"#)
            .unwrap();
        let v = r.get("values").unwrap();
        let on = ["a", "b", "c", "d"]
            .iter()
            .filter(|n| v.get(n).unwrap().as_f64() == Some(1.0))
            .count();
        assert!(on >= 3, "the floor must hold against an objective pushing the other way, got {on}");
    }

    #[test]
    fn the_slack_costs_spins_and_says_nothing() {
        // A caller should see the price in `spins` and never see the slack among the values.
        let plain = go(r#"{"variables":[{"name":"a","values":2},{"name":"b","values":2}],"tries":4}"#)
            .unwrap();
        let ineq = go(r#"{"variables":[{"name":"a","values":2},{"name":"b","values":2}],
                          "constraints":[{"type":"at_most","k":1,"of":[
                            {"var":"a","value":1},{"var":"b","value":1}]}],"tries":4}"#)
            .unwrap();
        assert!(
            ineq.get("spins").unwrap().as_f64() > plain.get("spins").unwrap().as_f64(),
            "an inequality costs spins"
        );
        let keys: Vec<&String> = match ineq.get("values").unwrap() {
            Json::Obj(m) => m.iter().map(|(k, _)| k).collect(),
            _ => Vec::new(),
        };
        assert_eq!(keys.len(), 2, "and reports only the caller's variables: {keys:?}");
    }
}

#[cfg(test)]
mod silent_wrongness {
    //! Four ways this surface used to answer a different question than the one asked.
    //!
    //! Every one of them returned `feasible: true` with a confident answer and no error, which is
    //! the only kind of bug a caller cannot defend against. They are grouped here because they
    //! share a cause: a reader that could not understand its input quietly substituted a default.

    use super::*;

    fn go(body: &str) -> Result<Json, String> {
        dispatch("solve", &crate::json::parse(body).unwrap())
    }

    fn value_of_x(body: &str) -> Option<f64> {
        go(body).ok()?.get("values")?.get("x")?.as_f64()
    }

    #[test]
    fn an_objective_term_can_name_any_number_of_variables() {
        // "these three together", which this surface could not say: it offered one variable or a
        // pair and nothing wider.
        let r = dispatch("solve", &crate::json::parse(
            r#"{"variables":[{"name":"a","values":3},{"name":"b","values":3},
                             {"name":"c","values":3}],
                 "objective":{"maximize":true,"terms":[
                     {"weight":9,"of":[{"var":"a","value":2},{"var":"b","value":2},
                                       {"var":"c","value":2}]}]},
                 "tries":24}"#).unwrap()).unwrap();
        let v = r.get("values").unwrap();
        for k in ["a", "b", "c"] {
            assert_eq!(v.get(k).unwrap().as_f64(), Some(2.0),
                       "the reward is only paid when all three hold ({k})");
        }
        assert!(r.get("ancillas").unwrap().as_f64().unwrap() > 0.0,
                "and the reply says what the lowering cost");
        assert_eq!(r.get("feasible").unwrap().as_bool(), Some(true));

        // a pairwise model reports none, so the field is a real signal rather than decoration
        let flat = dispatch("solve", &crate::json::parse(
            r#"{"variables":[{"name":"a","values":3},{"name":"b","values":3}],
                 "constraints":[{"type":"not_equal","a":"a","b":"b"}],"tries":8}"#).unwrap()).unwrap();
        assert_eq!(flat.get("ancillas").unwrap().as_f64(), Some(0.0));

        // and an empty product is refused rather than quietly ignored
        assert!(dispatch("solve", &crate::json::parse(
            r#"{"variables":[{"name":"a","values":3}],
                 "objective":{"maximize":true,"terms":[{"weight":1,"of":[]}]}}"#).unwrap()).is_err());
    }

    #[test]
    fn counting_constraints_take_any_number_of_entries() {
        // Nine of them, which the positional C form cannot express. The JSON surface always could;
        // this pins that it stays true.
        let vars: String = (0..9).map(|i| format!(r#"{{"name":"s{i}","values":2}}"#))
            .collect::<Vec<_>>().join(",");
        let of: String = (0..9).map(|i| format!(r#"{{"var":"s{i}","value":1}}"#))
            .collect::<Vec<_>>().join(",");
        let terms: String = (0..9).map(|i| format!(r#"{{"var":"s{i}","value":1,"weight":{}}}"#, 9 - i))
            .collect::<Vec<_>>().join(",");
        let r = dispatch("solve", &crate::json::parse(&format!(
            r#"{{"variables":[{vars}],
                 "constraints":[{{"type":"at_most","k":2,"of":[{of}]}}],
                 "objective":{{"maximize":true,"terms":[{terms}]}},"tries":16}}"#
        )).unwrap()).unwrap();
        assert_eq!(r.get("feasible").unwrap().as_bool(), Some(true));
        let v = r.get("values").unwrap();
        let on = (0..9).filter(|i| v.get(&format!("s{i}")).unwrap().as_f64() == Some(1.0)).count();
        assert_eq!(on, 2, "at most 2 of nine");
    }

    #[test]
    fn exactly_one_and_at_most_one_are_reachable_and_cheaper() {
        let body = |kind: &str| format!(
            r#"{{"variables":[{{"name":"a","values":2}},{{"name":"b","values":2}},
                              {{"name":"c","values":2}}],
                 "constraints":[{{"type":"{kind}","of":[{{"var":"a","value":1}},
                                                        {{"var":"b","value":1}},
                                                        {{"var":"c","value":1}}]}}],
                 "objective":{{"maximize":false,
                               "terms":[{{"var":"a","value":1,"weight":1}},
                                        {{"var":"b","value":1,"weight":1}},
                                        {{"var":"c","value":1,"weight":1}}]}},"tries":16}}"#
        );
        let count = |kind: &str| {
            let r = dispatch("solve", &crate::json::parse(&body(kind)).unwrap()).unwrap();
            let v = r.get("values").unwrap();
            let n = ["a", "b", "c"].iter()
                .filter(|k| v.get(k).unwrap().as_f64() == Some(1.0)).count();
            (n, r.get("spins").unwrap().as_f64().unwrap(),
             r.get("feasible").unwrap().as_bool() == Some(true))
        };
        // pushed off, at-most-one takes none and exactly-one still takes one
        assert_eq!(count("exactly_one"), (1, 6.0, true));
        assert_eq!(count("at_most_one"), (0, 6.0, true));

        // and neither costs a slack variable, where the inequality form does
        let r = dispatch("solve", &crate::json::parse(&body("at_most").replace(
            r#""type":"at_most""#, r#""type":"at_most","k":1"#)).unwrap()).unwrap();
        assert!(r.get("spins").unwrap().as_f64().unwrap() > 6.0,
                "at_most k=1 pays for a slack variable that at_most_one does not");

        let e = dispatch("solve", &crate::json::parse(
            r#"{"variables":[{"name":"a","values":2}],
                "constraints":[{"type":"nonsense","of":[]}]}"#).unwrap()).unwrap_err();
        assert!(e.contains("exactly_one") && e.contains("at_most_one"),
                "the known list must include them: {e}");
    }

    #[test]
    fn a_caller_can_lengthen_the_annealing_ladder() {
        let body = |sched: &str| format!(
            r#"{{"variables":[{{"name":"a","values":3}},{{"name":"b","values":3}}],
                 "constraints":[{{"type":"not_equal","a":"a","b":"a2"}}],"tries":4{sched}}}"#
        ).replace("\"a2\"", "\"b\"");

        // the default ladder still works with no schedule given
        let r = dispatch("solve", &crate::json::parse(&body("")).unwrap()).unwrap();
        assert_eq!(r.get("feasible").unwrap().as_bool(), Some(true));

        // and a caller's own ladder is accepted
        let r = dispatch("solve", &crate::json::parse(
            &body(r#","schedule":{"beta_hot":0.05,"beta_cold":6.0,"stages":60,"sweeps":20}"#)
        ).unwrap()).unwrap();
        assert_eq!(r.get("feasible").unwrap().as_bool(), Some(true));

        // partially given: the rest come from the default
        assert!(dispatch("solve", &crate::json::parse(
            &body(r#","schedule":{"stages":40}"#)
        ).unwrap()).is_ok());

        // a ladder that runs backwards is refused rather than silently substituted
        let e = dispatch("solve", &crate::json::parse(
            &body(r#","schedule":{"beta_hot":8.0,"beta_cold":0.05}"#)
        ).unwrap()).unwrap_err();
        assert!(e.contains("must exceed") && e.contains("hot to cold"), "{e}");
    }

    #[test]
    fn the_ceiling_counts_the_whole_run_not_just_the_burn_in() {
        // A ceiling that only looks at burn-in is not a ceiling. This request declared 1,024 node
        // updates at the gate and then did a quarter of a billion in the recording loop.
        let sneaky = r#"{"graph":{"builtin":"lattice2d","l":32},"sweeps":1,"draws":128,"thin":2000}"#;
        let cheap = r#"{"graph":{"builtin":"lattice2d","l":32},"sweeps":1,"draws":8,"thin":2}"#;

        // it is under the real ceiling, so it must still be ACCEPTED -- the point is the accounting
        let r = dispatch("sample", &crate::json::parse(sneaky).unwrap()).unwrap();
        let updates = r.get("ledger").and_then(|l| l.get("node_updates")).and_then(|x| x.as_f64());
        assert!(updates.unwrap() > 2.0e8, "the run really is that large: {updates:?}");

        // and a request past the ceiling is refused on the recording loop alone
        let over = format!(
            r#"{{"graph":{{"builtin":"lattice2d","l":64}},"sweeps":1,"draws":100000,"thin":100000}}"#
        );
        let e = dispatch("sample", &crate::json::parse(&over).unwrap()).unwrap_err();
        assert!(e.contains("draws") && e.contains("thin") && e.contains("ceiling"), "{e}");

        // the cheap one is unaffected
        assert!(dispatch("sample", &crate::json::parse(cheap).unwrap()).is_ok());

        // verify had no ceiling at all
        let e = dispatch("verify", &crate::json::parse(
            r#"{"graph":{"builtin":"ring","n":20},"draws":100000000,"thin":100000}"#
        ).unwrap()).unwrap_err();
        assert!(e.contains("ceiling"), "{e}");
    }

    #[test]
    fn maximize_as_a_number_maximizes() {
        // `as_bool` returned None for a JSON number, `unwrap_or(false)` made that Minimize, and the
        // caller got the OPPOSITE of what they asked for. Not a degraded answer -- the other one.
        let terms = r#"[{"var":"x","value":0,"weight":1},{"var":"x","value":4,"weight":5}]"#;
        let req = |m: &str| format!(
            r#"{{"variables":[{{"name":"x","values":5}}],"objective":{{"maximize":{m},"terms":{terms}}},"tries":10}}"#
        );
        assert_eq!(value_of_x(&req("true")), Some(4.0), "the plain form");
        assert_eq!(value_of_x(&req("1")), Some(4.0), "and the integer form agrees with it");
        assert_eq!(value_of_x(&req("false")), value_of_x(&req("0")), "as do both false forms");
        assert_ne!(value_of_x(&req("true")), value_of_x(&req("false")), "and the two differ");

        // anything that is neither is refused rather than read as false
        let e = go(&req(r#""yes""#)).unwrap_err();
        assert!(e.contains("must be true or false") && e.contains("a string"), "{e}");
    }

    #[test]
    fn a_value_that_is_not_a_number_is_refused() {
        // `"13"` -- what a shell pipeline or a templating layer produces -- used to become 0.
        let e = go(r#"{"variables":[{"name":"x","values":20}],
                       "constraints":[{"type":"fix","var":"x","value":"13"}],"tries":4}"#).unwrap_err();
        assert!(e.contains("whole number") && e.contains("a string"), "{e}");

        // and the number still works, so the check is not just refusing everything
        assert_eq!(value_of_x(r#"{"variables":[{"name":"x","values":20}],
                       "constraints":[{"type":"fix","var":"x","value":13}],"tries":4}"#), Some(13.0));

        // every place a value is read, not just `fix`
        for body in [
            r#"{"variables":[{"name":"x","values":3},{"name":"y","values":3}],
                "constraints":[{"type":"at_most","k":1,"of":[{"var":"x","value":"1"},{"var":"y","value":1}]}]}"#,
            r#"{"variables":[{"name":"x","values":3},{"name":"y","values":3}],
                "constraints":[{"type":"cardinality","k":1,"of":[{"var":"x","value":1},{"var":"y","value":[]}]}]}"#,
            r#"{"variables":[{"name":"x","values":3}],
                "objective":{"maximize":true,"terms":[{"var":"x","value":"2","weight":1}]}}"#,
        ] {
            assert!(go(body).is_err(), "a string value should be refused here: {body}");
        }
    }

    #[test]
    fn a_variable_cannot_be_compared_with_itself() {
        // The C ABI refuses this. The JSON surface accepted it and returned a feasible answer to a
        // request nothing can satisfy.
        for kind in ["not_equal", "equal"] {
            let e = go(&format!(
                r#"{{"variables":[{{"name":"a","values":3}}],
                     "constraints":[{{"type":"{kind}","a":"a","b":"a"}}],"tries":4}}"#
            ));
            let e = e.unwrap_err();
            assert!(e.contains("cannot be compared with itself") && e.contains('a'), "{kind}: {e}");
        }
    }

    #[test]
    fn a_boundary_inequality_is_not_dropped() {
        // `at most 0` needs no slack, and "needs no slack" was taken to mean "needs no constraint".
        let r = go(r#"{"variables":[{"name":"a","values":2},{"name":"b","values":2}],
             "constraints":[{"type":"at_most","k":0,"of":[{"var":"a","value":1},{"var":"b","value":1}]}],
             "objective":{"maximize":true,
                          "terms":[{"var":"a","value":1,"weight":1},{"var":"b","value":1,"weight":1}]},
             "tries":12}"#).unwrap();
        let v = r.get("values").unwrap();
        assert_eq!(v.get("a").unwrap().as_f64(), Some(0.0), "at most 0 means none");
        assert_eq!(v.get("b").unwrap().as_f64(), Some(0.0), "even against a reward on both");
        assert_eq!(r.get("feasible").unwrap().as_bool(), Some(true));

        // the mirror boundary: at least all of them
        let r = go(r#"{"variables":[{"name":"a","values":2},{"name":"b","values":2}],
             "constraints":[{"type":"at_least","k":2,"of":[{"var":"a","value":1},{"var":"b","value":1}]}],
             "objective":{"maximize":false,
                          "terms":[{"var":"a","value":1,"weight":1},{"var":"b","value":1,"weight":1}]},
             "tries":12}"#).unwrap();
        let v = r.get("values").unwrap();
        assert_eq!((v.get("a").unwrap().as_f64(), v.get("b").unwrap().as_f64()),
                   (Some(1.0), Some(1.0)), "at least 2 of 2 means both, against a penalty on both");
    }
}
