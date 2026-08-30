//! The operations, once, so that HTTP and MCP cannot drift apart.
//!
//! Every handler takes a JSON request and returns a JSON result or an error string. The transports
//! in `http` and `mcp` do nothing but move bytes and shape envelopes.

use crate::json::{parse, Json};
use ferrotherm::encode::Encoding;
use ferrotherm::gibbs::Sampler;
use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::ising;
use ferrotherm::certify::{certify, Certificate};
use ferrotherm::model::{Constraint, Expr, Lit, Model, Rel, Sense};
use ferrotherm::ledger::{Ledger, Prices, Z1_SPICE};
use std::time::Instant;

/// Ceilings that keep one request from monopolising the process. They are advertised in
/// `capabilities` so a caller can size a job instead of discovering the wall by hitting it.
pub const MAX_NODES: usize = 4_000_000;

/// Couplings a single request may compile to.
///
/// Separate from [`MAX_NODES`] because they are different dimensions and only one of them bounded
/// anything: a one-hot variable over k values is k spins and k(k-1)/2 couplings, so a request can
/// sit far under the node ceiling while the coupling count grows as the square of a number in the
/// request body.
///
/// The number is measured, not guessed. Cost is linear in couplings at about 34 bytes of reply and
/// 13 microseconds each on this machine:
///
/// | one-hot k | couplings | reply | wall |
/// |---|---|---|---|
/// | 100 | 4,950 | 161 KB | 0.08 s |
/// | 300 | 44,850 | 1.5 MB | 0.58 s |
/// | 600 | 179,700 | 6.2 MB | 2.38 s |
/// | 1000 | 499,500 | **17 MB** | **6.73 s** |
///
/// 100,000 holds one request to roughly 3.4 MB and 1.3 s, and still admits a one-hot over 447
/// values -- far past any model anyone writes by hand.
pub const MAX_COUPLINGS: usize = 100_000;
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

/// Base64, hand-rolled: this crate has no dependencies and one encoder does not justify the first.
fn b64(bytes: &[u8]) -> String {
    const A: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let b = [c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(A[(n >> 18 & 63) as usize] as char);
        out.push(A[(n >> 12 & 63) as usize] as char);
        out.push(if c.len() > 1 { A[(n >> 6 & 63) as usize] as char } else { '=' });
        out.push(if c.len() > 2 { A[(n & 63) as usize] as char } else { '=' });
    }
    out
}

fn ledger_json(l: &Ledger, p: &Prices, wall_s: f64) -> Json {
    // Counts are exact and always reported: they are what this run actually did. Joules are a
    // PROJECTION onto a device model, so they are reported only when that model states prices, and
    // the note is generated from the prices rather than written here -- a hardcoded note is how a
    // figure ends up labelled with the wrong machine when the prices change underneath it.
    let (fs, fr, fw) = l.shares(p);
    let num = |v: f64| if v.is_finite() { Json::n(v) } else { Json::Null };
    Json::obj(vec![
        ("node_updates", Json::n(l.samples as f64)),
        ("reads", Json::n(l.reads as f64)),
        ("writes", Json::n(l.writes as f64)),
        // Named for what it is: a projection onto a named device, null when that device has none.
        (
            "joules",
            match l.joules(p) {
                Some(j) => Json::n(j),
                None => Json::Null,
            },
        ),
        ("priced_as", Json::s(p.source)),
        (
            "share",
            Json::obj(vec![("sample", num(fs)), ("read", num(fr)), ("write", num(fw))]),
        ),
        ("wall_seconds", Json::n(wall_s)),
        (
            "note",
            Json::s(&format!(
                "Counts are what this run did and are exact. \"joules\" is a projection of those \
                 counts onto a device model, null when the model publishes no per-operation \
                 energy; \"priced_as\" says whose numbers they are. {}",
                p.source
            )),
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
    // The library's ladder, not a second opinion.
    //
    // These were 0.1 -> 3.0 over 24 x 20 = 480 sweeps, where Python (__init__.py:334), Julia
    // (Ferrotherm.jl:418) and the browser IDE (docs/ide.html) all use 0.05 -> 4.0 over 60 x 40 =
    // 2400. Three surfaces agreeing and one disagreeing is not a design choice, it is drift: the
    // same "use the defaults" request returned a worse best_energy here than everywhere else, on
    // the same core, with nothing to say why.
    let beta_min = opt_f64(req, "beta_min", 0.05);
    let beta_max = opt_f64(req, "beta_max", 4.0);
    // Clamped, matching what /v1/solve's schedule arm already does. Unbounded, `stages` came
    // straight from the request body into an allocation.
    let stages = opt_usize(req, "stages", 60).clamp(2, 10_000);
    let per = opt_usize(req, "sweeps_per_stage", 40).clamp(1, 100_000);
    let seed = req.get("seed").and_then(|s| s.as_u64()).unwrap_or(0);
    if !(beta_min > 0.0 && beta_max > beta_min) {
        return Err("need 0 < beta_min < beta_max".into());
    }
    // Saturating in u64 the whole way. `(stages * per) as u64` multiplies in USIZE first, so it
    // wrapped before the cast ever happened: `stages = 2^63` gave a small budget, passed this
    // ceiling, and aborted the server in `raw_vec` with a capacity overflow -- an empty reply
    // rather than a 400, and the process gone.
    let budget = (g.n as u64)
        .saturating_mul(stages as u64)
        .saturating_mul(per as u64);
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

/// Every lower bound on the ground energy, and — if a state is supplied — how far it is from
/// optimal.
///
/// THE OPERATION THIS SERVER WAS MISSING. `verify` checks the sampler against the exact
/// distribution and is capped at 20 nodes because it enumerates; this answers the same question —
/// *should I believe this answer* — at any size, and answers it about the OPTIMUM rather than the
/// distribution. `energy − best` is an upper limit on what a better search could still win, and
/// zero means the state is proved optimal without trusting whatever produced it.
///
/// All four bounds are sound on their own, so `best` is their maximum. They disagree by a lot and
/// in both directions: `odd_cycle` wins on sparse frustrated lattices, `sdp` by more the denser the
/// instance gets, and `forest` is worth nothing at all on a graph with no fields.
pub fn bound(req: &Json) -> Result<Json, String> {
    let g = graph_from(req.get("graph").ok_or("missing \"graph\"")?)?;
    let rounds = opt_usize(req, "forest_rounds", 40).clamp(0, 10_000);
    let max_cycle = opt_usize(req, "max_cycle", 6).clamp(0, 64);
    let sdp_sweeps = opt_usize(req, "sdp_sweeps", 200).clamp(1, 100_000);
    let seed = req.get("seed").and_then(|s| s.as_u64()).unwrap_or(1);

    // The SDP factors a dense n x n matrix, so its cost is n^3 in the NODE COUNT and does not care
    // how sparse the graph is. Refused rather than attempted: an unbounded Cholesky here is a way
    // to take the server down with a valid-looking request.
    const SDP_MAX_NODES: usize = 2_048;
    let sdp = if g.n <= SDP_MAX_NODES {
        let p = ferrotherm::sdp::Params { sweeps: sdp_sweeps, ..ferrotherm::sdp::Params::default() };
        let (_, cert) = ferrotherm::sdp::certified(&g, &p, seed);
        // Re-verified here, not trusted: rebuild the cost matrix from the graph and re-run the
        // positive-definiteness proof before the number leaves this process.
        cert.verify(&g).ok()
    } else {
        None
    };

    let d = ferrotherm::bound::decoupled(&g).value;
    let f = ferrotherm::bound::forest(&g, rounds).value;
    let c = ferrotherm::bound::odd_cycle(&g, max_cycle).value;
    let mut pairs = vec![("decoupled", d), ("forest", f), ("odd_cycle", c)];
    if let Some(v) = sdp {
        pairs.push(("sdp", v));
    }
    let (which, best) = pairs
        .iter()
        .fold(("decoupled", f64::NEG_INFINITY), |acc, &(n, v)| if v > acc.1 { (n, v) } else { acc });

    let mut out = vec![
        ("nodes", Json::n(g.n as f64)),
        ("decoupled", Json::n(d)),
        ("forest", Json::n(f)),
        ("odd_cycle", Json::n(c)),
        (
            "sdp",
            match sdp {
                Some(v) => Json::n(v),
                None if g.n > SDP_MAX_NODES => {
                    Json::s("refused: the SDP is O(n^3) dense and this graph is over 2048 nodes")
                }
                None => Json::s("refused: the certificate did not re-verify"),
            },
        ),
        ("best", Json::n(best)),
        ("which", Json::s(which)),
    ];

    if let Some(sv) = req.get("state").and_then(|s| s.as_arr()) {
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
        let e = g.energy(&s);
        out.push(("energy", Json::n(e)));
        out.push(("gap", Json::n(e - best)));
    }
    Ok(Json::Obj(out.into_iter().map(|(k, v)| (k.to_string(), v)).collect()))
}

/// **Exact** max-cut on a planar graph, in polynomial time.
///
/// The only operation here that returns an OPTIMUM rather than an attempt. Max-cut is NP-hard in
/// general and polynomial on a planar graph, and the difference is a theorem: a cut in the graph is
/// a cycle in the dual, so the problem becomes a minimum-weight T-join and then a minimum-weight
/// perfect matching. There is no budget and no seed — the same request always returns the same
/// answer, because there is only one.
pub fn exact_planar(req: &Json) -> Result<Json, String> {
    let g = graph_from(req.get("graph").ok_or("missing \"graph\"")?)?;
    let scale = opt_f64(req, "scale", 1.0);
    if !(scale.is_finite() && scale > 0.0) {
        return Err("\"scale\" must be finite and positive".into());
    }
    // The matching underneath is O(k^3) in the odd-degree dual vertices, which is O(n) — so this is
    // cubic in the node count and needs a ceiling like everything else here.
    const MAX_NODES: usize = 20_000;
    if g.n > MAX_NODES {
        return Err(format!(
            "{} nodes, over the {MAX_NODES} ceiling for the exact planar solver: the matching \
             underneath is cubic",
            g.n
        ));
    }
    let t0 = Instant::now();
    let out = ferrotherm::planarcut::solve(&g, &ferrotherm::planarcut::Params { scale })
        // The four refusals are four different things to do next, so the reason is the reply.
        .map_err(|e| e.to_string())?;
    let wall = t0.elapsed().as_secs_f64();
    let mut fields = vec![
        ("nodes", Json::n(g.n as f64)),
        ("cut", Json::n(out.cut)),
        ("energy", Json::n(out.energy)),
        ("exact", Json::Bool(true)),
        ("faces", Json::n(out.faces as f64)),
        // The size of the matching problem, and the real cost driver. Zero is legitimate.
        ("odd_faces", Json::n(out.odd_faces as f64)),
        ("wall_seconds", Json::n(wall)),
    ];
    if req.get("return_state").and_then(|b| b.as_bool()).unwrap_or(g.n <= 4096) {
        fields.push(("state", state_json(&out.state)));
    } else {
        fields.push(("state_omitted", Json::s("pass \"return_state\": true to include it")));
    }
    Ok(Json::Obj(fields.into_iter().map(|(k, v)| (k.to_string(), v)).collect()))
}

/// An **upper bound** on the maximum cut of a toroidal grid.
///
/// The side of the G-set table nobody publishes: every figure there is a best cut *found*, a lower
/// bound. This is the other end of the bracket. On G11 it closes it, proving the twenty-five-year-old
/// best-known cut of 564 optimal.
pub fn toroidal_bound(req: &Json) -> Result<Json, String> {
    let g = graph_from(req.get("graph").ok_or("missing \"graph\"")?)?;
    let scale = opt_f64(req, "scale", 1.0);
    if !(scale.is_finite() && scale > 0.0) {
        return Err("\"scale\" must be finite and positive".into());
    }
    const MAX_NODES: usize = 20_000;
    if g.n > MAX_NODES {
        return Err(format!("{} nodes, over the {MAX_NODES} ceiling: the matching is cubic", g.n));
    }
    let emb = ferrotherm::planar::torus_grid_of(&g).ok_or(
        "not a toroidal grid. The structure is recovered from the edge list -- a match on all 2n \
         edges -- rather than assumed, so this is a statement about the graph, not a limitation",
    )?;
    let t0 = Instant::now();
    let b = ferrotherm::planarcut::bound_on_surface(
        &g,
        &emb,
        &ferrotherm::planarcut::Params { scale },
    )
    .map_err(|e| e.to_string())?;
    let wall = t0.elapsed().as_secs_f64();
    let mut fields = vec![
        ("nodes", Json::n(g.n as f64)),
        ("upper_bound", Json::n(b.cut)),
        // ATTAINED means the bound is the maximum, proved. Not attained leaves it a bound, and
        // conflating the two is the only way to misuse this number.
        ("attained", Json::Bool(b.attained)),
        ("genus", Json::n(b.genus as f64)),
        ("faces", Json::n(b.faces as f64)),
        ("odd_faces", Json::n(b.odd_faces as f64)),
        ("wall_seconds", Json::n(wall)),
    ];
    if let (Some(s), Some(e)) = (&b.state, b.energy) {
        fields.push(("energy", Json::n(e)));
        if req.get("return_state").and_then(|v| v.as_bool()).unwrap_or(g.n <= 4096) {
            fields.push(("state", state_json(s)));
        }
    }
    Ok(Json::Obj(fields.into_iter().map(|(k, v)| (k.to_string(), v)).collect()))
}

/// Identity on a state, so every arm of `optimize` returns the same shape.
fn o_state(s: Vec<i8>) -> Vec<i8> {
    s
}

/// Minimise by a named method: tabu search, population annealing, or branch and bound.
///
/// `anneal` runs one chain down a ladder and hands back the best state it saw. These are the three
/// things that do more than that: `tabu` remembers where it has been, `population` reports whether
/// its own answer is trustworthy, and `branch` returns a PROOF or says it has none.
pub fn optimize(req: &Json) -> Result<Json, String> {
    let g = graph_from(req.get("graph").ok_or("missing \"graph\"")?)?;
    let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("tabu").to_string();
    let seed = req.get("seed").and_then(|s| s.as_u64()).unwrap_or(0);
    let t0 = Instant::now();
    let mut led = Ledger::default();

    let (state, mut extra): (Vec<i8>, Vec<(&str, Json)>) = match method.as_str() {
        "tabu" => {
            let iterations = opt_usize(req, "iterations", 50_000).clamp(1, 100_000_000);
            let tenure = opt_usize(req, "tenure", 0).min(g.n);
            let restart = opt_usize(req, "restart_after", 5_000);
            // A tabu iteration evaluates every node's move, so the honest budget is n per
            // iteration -- the same accounting the ledger uses.
            let budget = (g.n as u64).saturating_mul(iterations as u64);
            if budget > MAX_NODE_UPDATES {
                return Err(format!(
                    "{budget} move evaluations requested, over the {MAX_NODE_UPDATES} ceiling"
                ));
            }
            let p = ferrotherm::tabu::Params {
                iterations,
                tenure,
                restart_after: (restart > 0).then_some(restart),
                // The same "incumbent" branch has always taken. Tabu could not accept one until
                // now, so a caller with a good state had to run tabu FIRST or not at all.
                start: start_from(req, g.n)?,
            };
            let o = ferrotherm::tabu::search_metered(&g, &p, seed, Some(&mut led));
            let ran = o.iterations_run;
            (
                o.state,
                vec![
                    ("iterations_requested", Json::n(iterations as f64)),
                    // Reported because truncation is otherwise invisible: a run that spent a
                    // fraction of its budget returns a result shaped exactly like a full one.
                    ("iterations_run", Json::n(ran as f64)),
                    ("restarts", Json::n(o.restarts as f64)),
                ],
            )
        }
        "breakout" => {
            let iterations = opt_usize(req, "iterations", 50_000).clamp(1, 100_000_000);
            // One iteration is one FLIP here, and the ledger charges n move evaluations for it --
            // two different numbers measuring two different things. The ceiling is on the
            // evaluations, because that is the work.
            let budget = (g.n as u64).saturating_mul(iterations as u64);
            if budget > MAX_NODE_UPDATES {
                return Err(format!(
                    "{budget} move evaluations requested, over the {MAX_NODE_UPDATES} ceiling"
                ));
            }
            let p = ferrotherm::bls::Params {
                iterations,
                start: start_from(req, g.n)?,
                ..ferrotherm::bls::Params::default()
            };
            let o = ferrotherm::bls::search_metered(&g, &p, seed, Some(&mut led));
            let pert = o.perturbations;
            (
                o.state,
                vec![
                    ("iterations_requested", Json::n(iterations as f64)),
                    ("iterations_run", Json::n(o.iterations_run as f64)),
                    // The claim BLS makes is about what happens BETWEEN local optima, so a run with
                    // a handful of descents did not run the algorithm -- and the energy alone would
                    // not say so.
                    ("descents", Json::n(o.descents as f64)),
                    ("max_jump", Json::n(o.max_jump as f64)),
                    ("returns_to_previous_optimum", Json::n(o.returns as f64)),
                    (
                        "perturbations",
                        Json::obj(vec![
                            ("directed_one", Json::n(pert.directed_one as f64)),
                            ("directed_two", Json::n(pert.directed_two as f64)),
                            ("random", Json::n(pert.random as f64)),
                        ]),
                    ),
                ],
            )
        }
        "cluster" => {
            // Parallel tempering with isoenergetic cluster moves: the baseline the field measures
            // against. Two ladders, so every temperature has a partner to exchange a cluster with.
            let rungs = opt_usize(req, "rungs", 16).clamp(2, 512);
            let rounds = opt_usize(req, "rounds", 400).clamp(1, 1_000_000);
            let bmin = opt_f64(req, "beta_min", 0.1);
            let bmax = opt_f64(req, "beta_max", 6.0);
            if !(bmin > 0.0 && bmax > bmin) {
                return Err("need 0 < beta_min < beta_max".into());
            }
            // Two ladders of `rungs` replicas, each doing `rounds` sweeps.
            let budget = (g.n as u64)
                .saturating_mul(rungs as u64)
                .saturating_mul(rounds as u64)
                .saturating_mul(2);
            if budget > MAX_NODE_UPDATES {
                return Err(format!(
                    "{budget} node updates requested, over the {MAX_NODE_UPDATES} ceiling"
                ));
            }
            let p = ferrotherm::icm::Params {
                betas: ferrotherm::tempering::geometric_ladder(bmin, bmax, rungs),
                rounds,
                sweeps_per_round: 1,
                swap_every: 1,
                icm_every: 1,
            };
            let o = ferrotherm::icm::run_metered(&g, &p, seed, Some(&mut led))
                // A field breaks the isoenergetic argument; the reason is the reply.
                .map_err(|e| e.to_string())?;
            (
                o.state,
                vec![
                    ("rungs", Json::n(rungs as f64)),
                    ("rounds", Json::n(rounds as f64)),
                    // A move that never fires is not a move: two replicas that agree everywhere
                    // have no disagreement subgraph and nothing to exchange.
                    ("cluster_moves", Json::n(o.icm_moves as f64)),
                    (
                        "mean_cluster_size",
                        Json::n(if o.icm_moves == 0 { 0.0 } else { o.icm_spins as f64 / o.icm_moves as f64 }),
                    ),
                ],
            )
        }
        "quantum" => {
            // Path-integral Monte Carlo on the transverse-field Ising model. NOT a quantum
            // computer: the word describes what is modelled, not what runs.
            let trotter = opt_usize(req, "trotter", 4).clamp(1, 1024);
            let beta = opt_f64(req, "beta", 10.0);
            let gmax = opt_f64(req, "gamma_max", 3.0);
            let gmin = opt_f64(req, "gamma_min", 0.05);
            let steps = opt_usize(req, "steps", 200).clamp(1, 1_000_000);
            if !(beta > 0.0 && gmax >= gmin && gmin >= 0.0) {
                return Err("need beta > 0 and gamma_max >= gamma_min >= 0".into());
            }
            let budget = (g.n as u64)
                .saturating_mul(trotter as u64)
                .saturating_mul(steps as u64);
            if budget > MAX_NODE_UPDATES {
                return Err(format!(
                    "{budget} spin proposals requested, over the {MAX_NODE_UPDATES} ceiling"
                ));
            }
            let p = ferrotherm::sqa::Params {
                trotter,
                beta,
                gamma_max: gmax,
                gamma_min: gmin,
                steps,
                sweeps_per_step: 1,
            };
            let o = ferrotherm::sqa::run_metered(&g, &p, seed, Some(&mut led));
            (
                o.state,
                vec![
                    ("trotter", Json::n(trotter as f64)),
                    ("proposals", Json::n(o.proposals as f64)),
                    ("accepted", Json::n(o.accepted as f64)),
                    // The quantity that diverges if gamma_min is set to zero, so a reader can see
                    // how close the schedule got to the classical limit.
                    ("max_j_perp", Json::n(o.max_j_perp)),
                    (
                        "note",
                        Json::s(if trotter == 1 {
                            "one Trotter slice: this is CLASSICAL annealing, the control"
                        } else {
                            "path-integral Monte Carlo; a classical state, no quantum claim"
                        }),
                    ),
                ],
            )
        }
        "goemans_williamson" => {
            // The relaxation from the PRIMAL side. `optimize` returns a state, so this belongs
            // here rather than beside the bounds.
            let hyperplanes = opt_usize(req, "hyperplanes", 64).clamp(1, 100_000);
            const SDP_MAX_NODES: usize = 2_048;
            if g.n > SDP_MAX_NODES {
                return Err(format!(
                    "{} nodes, over the {SDP_MAX_NODES} ceiling: the relaxation is O(n^3) dense",
                    g.n
                ));
            }
            let r = ferrotherm::sdp::goemans_williamson(
                &g,
                &ferrotherm::sdp::Params::default(),
                seed,
                hyperplanes,
            );
            (
                o_state(r.state),
                vec![
                    ("cut", Json::n(r.cut)),
                    ("hyperplanes", Json::n(r.hyperplanes as f64)),
                    // FALSE on most instances people care about. A guarantee field that were
                    // always true would be a lie in the shape of a guarantee.
                    ("guaranteed", Json::Bool(r.guaranteed)),
                    (
                        "guarantee_note",
                        Json::s(if r.guaranteed {
                            "0.87856 of the SDP optimum in expectation: non-negative edge weights"
                        } else {
                            "no ratio applies: the 0.87856 bound needs non-negative edge weights, \
                             which means non-positive couplings and no fields"
                        }),
                    ),
                ],
            )
        }
        "population" => {
            let population = opt_usize(req, "population", 1_000).clamp(1, 1_000_000);
            let sweeps = opt_usize(req, "sweeps", 4).clamp(1, 100_000);
            let stages = opt_usize(req, "stages", 100).clamp(1, 100_000);
            let beta_max = opt_f64(req, "beta_max", 6.0);
            if !(beta_max.is_finite() && beta_max > 0.0) {
                return Err("need beta_max > 0".into());
            }
            let budget = (g.n as u64)
                .saturating_mul(population as u64)
                .saturating_mul(sweeps as u64)
                .saturating_mul(stages as u64 + 1);
            if budget > MAX_NODE_UPDATES {
                return Err(format!(
                    "{budget} node updates requested, over the {MAX_NODE_UPDATES} ceiling"
                ));
            }
            let p = ferrotherm::popanneal::Params::linear_from_zero(
                population, sweeps, beta_max, stages,
            );
            let o = ferrotherm::popanneal::run(&g, &p, seed);
            (
                o.state,
                vec![
                    ("population", Json::n(population as f64)),
                    ("beta_max", Json::n(beta_max)),
                    (
                        "ln_z",
                        if o.ln_z_is_absolute { Json::n(o.ln_z) } else { Json::Null },
                    ),
                    // THE NUMBER THAT SAYS WHETHER TO BELIEVE ln_z. 1 is ideal; the population size
                    // means every survivor descends from one ancestor.
                    ("rho", Json::n(o.rho_max)),
                    (
                        "rho_reading",
                        Json::s(if o.rho_max <= (population as f64 / 10.0).max(1.0) {
                            "the population kept its diversity"
                        } else {
                            "the population collapsed; ln_z is not trustworthy"
                        }),
                    ),
                ],
            )
        }
        "branch" => {
            let max_nodes = req
                .get("max_nodes")
                .and_then(|v| v.as_u64())
                .unwrap_or(20_000_000)
                .clamp(1, 2_000_000_000);
            let incumbent = req.get("incumbent").and_then(|s| s.as_arr()).map(|sv| {
                sv.iter().map(|x| if x.as_f64() == Some(1.0) { 1i8 } else { -1 }).collect::<Vec<i8>>()
            });
            // `sdp_depth` is exposed because it is the one dial whose value is instance-dependent:
            // it buys a much tighter bound at the top of the tree and costs a Cholesky per node to
            // do it, and which way that lands depends on the density of the graph in the request.
            let sdp_depth = req.get("sdp_depth").and_then(|v| v.as_usize()).filter(|d| *d <= 16);
            let p = ferrotherm::branch::Params {
                max_nodes,
                incumbent,
                sdp_depth,
                ..ferrotherm::branch::Params::default()
            };
            let o = ferrotherm::branch::solve(&g, &p);
            (
                o.state,
                vec![
                    // True ONLY when the tree was exhausted. A run that hit the budget returns its
                    // best state and says the proof is missing.
                    ("proved_optimal", Json::Bool(o.proved_optimal)),
                    ("nodes", Json::n(o.nodes as f64)),
                    ("pruned", Json::n(o.pruned as f64)),
                    ("hit_limit", Json::Bool(o.hit_limit)),
                    // Calls AND prunes. Reporting only the calls would say the bound RAN, which is
                    // not the same as saying it helped -- and a bound that fires a hundred times
                    // and cuts nothing is a hundred wasted Choleskys.
                    ("sdp_calls", Json::n(o.sdp_calls as f64)),
                    ("sdp_prunes", Json::n(o.sdp_prunes as f64)),
                ],
            )
        }
        other => {
            return Err(format!(
                "unknown method {other:?}; expected \"tabu\", \"breakout\", \"cluster\", \
                 \"quantum\", \"goemans_williamson\", \"population\" or \"branch\""
            ))
        }
    };

    let wall = t0.elapsed().as_secs_f64();
    // Recomputed from the state, not carried: the one number a caller acts on should not depend on
    // an accumulator inside the solver being right.
    let e = g.energy(&state);
    let mut out = vec![
        ("method", Json::s(&method)),
        ("nodes_in_graph", Json::n(g.n as f64)),
        ("best_energy", Json::n(e)),
        ("seed", Json::n(seed as f64)),
        ("ledger", ledger_json(&led, &Z1_SPICE, wall)),
    ];
    out.append(&mut extra);
    if req.get("return_state").and_then(|b| b.as_bool()).unwrap_or(g.n <= 4096) {
        out.push(("state", state_json(&state)));
    } else {
        out.push(("state_omitted", Json::s("pass \"return_state\": true to include it")));
    }
    Ok(Json::Obj(out.into_iter().map(|(k, v)| (k.to_string(), v)).collect()))
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
                op("bound", "Lower bounds on the ground energy, and the gap of a supplied state. Any size.", "graph, state, forest_rounds, max_cycle, sdp_sweeps, seed"),
                op("exact_planar", "EXACT max-cut on a planar graph, in polynomial time. Not a search: it returns the maximum, not the best found.", "graph, scale, return_state"),
                op("toroidal_bound", "An UPPER bound on the maximum cut of a toroidal grid -- the side of the G-set table nobody publishes.", "graph, scale, return_state"),
                op("optimize", "Minimise by tabu search, breakout local search, isoenergetic cluster moves, simulated quantum annealing, Goemans-Williamson rounding, population annealing, or branch and bound.", "graph, method, seed, iterations, population, stages, beta_max, max_nodes, incumbent (branch, tabu and breakout all take one now), sdp_depth, return_state"),
                op("solve", "State a problem with named variables and constraints; get named values back. Constraint types: not_equal, equal, fix, cardinality, at_most, at_least, exactly_one, at_most_one, all_different, and linear -- a WEIGHTED row (3a + 4b + 5c <= 7), which no counting constraint can express. \"method\" chooses the solver -- anneal (default), tabu, breakout, or branch, which is the only one that PROVES its answer.", "variables, constraints, objective, tries, penalty, schedule, method, effort"),
                op("hubo", "Minimise a HIGHER-ORDER model natively -- terms of any arity, no ancillas and no penalty to get right. Use this rather than a three-or-more-variable objective term in \"solve\" whenever the target is a CPU.", "spins, terms, beta_min, beta_max, stages, sweeps_per_stage, seed"),
                op("fit", "FIT a Boltzmann machine to data and get the trained model back as a graph. The only operation here that produces a model rather than consuming one; its \"graph\" is the shape every other operation takes, so fit then sample, anneal, bound or verify with no export step.", "visible, hidden, data, epochs, k, positive_sweeps, learning_rate, batch, seed"),
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
        // How the variable is STORED. Silently ignoring this was the bug: a caller writing
        // "encoding": "binary" got one-hot, with a different spin count, a different penalty and no
        // error -- the reply's `ftp` said `onehot` and nothing else did. An unknown name is refused
        // by listing the ones that exist rather than falling back to a default.
        let encoding = match v.get("encoding") {
            None => Encoding::OneHot,
            Some(e) => {
                let s = e.as_str().ok_or_else(|| {
                    format!("{name}: \"encoding\" must be a string, not {}", describe(e))
                })?;
                match s {
                    "one-hot" | "onehot" => Encoding::OneHot,
                    "binary" | "log" => Encoding::Binary,
                    "domain-wall" | "domainwall" | "unary" => Encoding::DomainWall,
                    other => {
                        return Err(format!(
                            "{name}: unknown encoding {other:?}; known: one-hot (exact, k spins), \
                             domain-wall (exact, k-1 spins), binary (ceil(log2 k) spins, and NOT \
                             exact unless k is a power of two)"
                        ))
                    }
                }
            }
        };

        let h = match (v.get("values"), v.get("lo"), v.get("hi")) {
            (Some(k), _, _) => {
                let k = k.as_usize().ok_or_else(|| format!("{name}: \"values\" must be an integer"))?;
                if k < 2 {
                    return Err(format!("{name}: a variable with {k} values is a constant"));
                }
                m.categorical_as(&name, k, encoding)
            }
            (None, Some(lo), Some(hi)) => {
                let (lo, hi) = (
                    lo.as_f64().ok_or("\"lo\" must be a number")? as i64,
                    hi.as_f64().ok_or("\"hi\" must be a number")? as i64,
                );
                if hi <= lo {
                    return Err(format!("{name}: an integer range needs hi > lo"));
                }
                m.integer_as(&name, lo, hi, encoding)
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

            // A "soft" price turns any constraint into a preference the solver may trade away. It
            // is read here rather than in each arm so that every constraint kind gets it, including
            // ones added later -- the parity rule this surface has broken before.
            //
            // Not `.and_then(as_f64)`: a "soft" the reader cannot understand would silently become
            // a HARD constraint, which is the opposite of what was asked for, returned with no
            // error. A string "5" is a modelling mistake and says so.
            let soft = match c.get("soft") {
                None => None,
                Some(x) => Some(x.as_f64().ok_or_else(|| {
                    format!(
                        "constraint {i} ({kind}): \"soft\" is a price and must be a number, not {}. \
                         Omit it for a hard constraint.",
                        describe(x)
                    )
                })?),
            };
            if let Some(w) = soft {
                if !w.is_finite() || w <= 0.0 {
                    return Err(format!(
                        "constraint {i} ({kind}): \"soft\" must be a positive, finite price, not \
                         {w}. A price of zero or less is not a preference -- omit the field to make \
                         the constraint hard."
                    ));
                }
            }

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
                "all_different" => {
                    // Takes "of": [{"var": name}, ...] -- variables, not literals. A "value" here
                    // would be meaningless and is ignored rather than silently constraining
                    // something else.
                    let items = c.get("of").and_then(|x| x.as_arr())
                        .ok_or("all_different needs \"of\"")?;
                    if items.len() < 2 {
                        return Err("all_different needs at least two variables".into());
                    }
                    let mut vars = Vec::new();
                    for it in items {
                        let vn = it.get("var").and_then(|x| x.as_str()).ok_or(
                            "each entry in all_different's \"of\" needs a \"var\"",
                        )?;
                        let h = handles[find(vn)?];
                        if !vars.contains(&h) {
                            vars.push(h);
                        }
                    }
                    m.all_different(vars);
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
                "linear" => {
                    // A WEIGHTED row: {"type":"linear","of":[{"var":"a","coeff":3},...],
                    //                  "rel":"<=","rhs":7}
                    //
                    // The constraint none of the counting kinds above can express. They all count
                    // UNWEIGHTED literals, so `3a + 4b + 5c <= 7` could not be said here at all,
                    // and the only advice available was to add it to the objective -- which is not
                    // a constraint, so "feasible" and "violated" stop knowing about the row.
                    let rel = match c.get("rel").and_then(|x| x.as_str()).unwrap_or("<=") {
                        "<=" | "le" | "\u{2264}" => Rel::Le,
                        ">=" | "ge" | "\u{2265}" => Rel::Ge,
                        "=" | "==" | "eq" => Rel::Eq,
                        other => {
                            return Err(format!(
                                "linear: \"rel\" is \"<=\", \">=\" or \"=\", not {other:?}"
                            ))
                        }
                    };
                    let rhs = c
                        .get("rhs")
                        .and_then(|x| x.as_f64())
                        .ok_or("linear needs a numeric \"rhs\"")?;
                    let items =
                        c.get("of").and_then(|x| x.as_arr()).ok_or("linear needs \"of\"")?;
                    if items.is_empty() {
                        return Err("linear needs at least one term in \"of\"".into());
                    }
                    let mut terms = Vec::new();
                    for it in items {
                        let vn = it.get("var").and_then(|x| x.as_str()).unwrap_or("");
                        let vv = value_of(it, "value", 1, "linear")?;
                        // A missing coefficient is 1, so an unweighted row is still sayable and
                        // means what it looks like.
                        let w = match it.get("coeff") {
                            Some(x) => x.as_f64().ok_or_else(|| {
                                format!("linear: \"coeff\" on {vn:?} must be a number")
                            })?,
                            None => 1.0,
                        };
                        terms.push((Lit::Is(handles[find(vn)?], vv), w));
                    }
                    m.linear(terms, rel, rhs);
                }
                other => {
                    return Err(format!(
                        "unknown constraint {other:?}; known: not_equal, equal, fix, \
                         cardinality, at_most, at_least, exactly_one, at_most_one, \
                         all_different, linear"
                    ))
                }
            }

            if let Some(w) = soft {
                // Every arm above added exactly one constraint, so "the last one" is this one.
                if !m.soften_last(w) {
                    return Err(format!("constraint {i} ({kind}) could not be made soft"));
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
    // 12, matching Python (__init__.py:1205) and Julia (Ferrotherm.jl:1118). 16 here meant the
    // same request cost a third more anneals over HTTP than in-process, for no stated reason.
    let tries = opt_usize(req, "tries", 12).clamp(1, 500);

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
            Some(ferrotherm::schedule::Schedule::geometric(hot, cold, stages, per))
        }
    };

    // Outside the match, so the DEFAULT path is bounded by the same rule as the schedule one.
    // This check used to sit inside the schedule arm, so a request that named no schedule had no
    // ceiling at all: `{"variables":[{"name":"x","values":1000}],"tries":1}` -- 46 bytes -- built a
    // quadratically large graph, took 11.9 s and returned a 17 MB reply. Cost quadratic in a number
    // the caller types, with nothing between the two.
    if compiled.spins() > MAX_NODES {
        return Err(format!(
            "this model compiles to {} spins, over the {MAX_NODES} ceiling for one request",
            compiled.spins()
        ));
    }
    // The ceiling that actually binds here is COUPLINGS, not spins.
    //
    // `{"variables":[{"name":"x","values":1000}],"tries":1}` is 46 bytes and compiles to only 1000
    // spins -- comfortably under MAX_NODES, and one sweep, so no node-update bound fires either.
    // But a one-hot of k values carries k(k-1)/2 penalty couplings: 499,500 of them, which is what
    // made that request take 11.9 s and return a 17 MB reply. Cost quadratic in a number the caller
    // types, bounded by nothing, because every existing ceiling measured the wrong dimension.
    let couplings = compiled.program.factors.len();
    if couplings > MAX_COUPLINGS {
        return Err(format!(
            "this model compiles to {couplings} couplings, over the {MAX_COUPLINGS} ceiling for \
             one request. A one-hot variable over k values costs k(k-1)/2 couplings on its own, so \
             this grows as the square of a value you typed: use a smaller domain, or \
             Encoding::DomainWall, which is linear"
        ));
    }
    let sweeps_total = match &sched {
        // Sum the stages' own sweep counts rather than assuming a uniform ladder.
        Some(s) => s.stages().iter().fold(0usize, |a, st| a.saturating_add(st.sweeps)),
        None => 1,
    };
    bound_updates(compiled.spins(), sweeps_total.saturating_mul(tries), 0, 1)?;

    let t0 = Instant::now();
    // A method choice, defaulting to the anneal this always did. Only "branch" returns a proof, and
    // it is the reason this exists: the modelling layer -- the one every document here says to
    // reach for first -- was the one layer that could not certify its own answer.
    let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("anneal");
    let effort = req.get("effort").and_then(|v| v.as_usize()).unwrap_or(0);
    let chosen = match method {
        "anneal" => None,
        "tabu" => Some(ferrotherm::model::Method::Tabu {
            iterations: if effort == 0 { 50_000 } else { effort },
        }),
        "breakout" => Some(ferrotherm::model::Method::Breakout {
            iterations: if effort == 0 { 50_000 } else { effort },
        }),
        "branch" => Some(ferrotherm::model::Method::Branch {
            max_nodes: if effort == 0 { 20_000_000 } else { effort as u64 },
        }),
        other => {
            return Err(format!(
                "unknown method {other:?}; one of \"anneal\", \"tabu\", \"breakout\", \"branch\""
            ))
        }
    };
    let sol = match chosen {
        // A chosen method runs once: `tries` is the anneal's restart count and means nothing to a
        // deterministic search or to a proof.
        Some(m) => compiled.solve_by(m, 1),
        None => match &sched {
            Some(s) => compiled.solve_best_with(s, tries as u64),
            None => compiled.solve_best_of(tries as u64),
        },
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
                            // A hard one means the answer is not an answer. A soft one means the
                            // solver made the trade it was asked to price, and the answer stands.
                            // Both appear here; only the hard ones move "feasible".
                            ("hard", Json::Bool(v.hard)),
                            ("cost", Json::n(v.cost)),
                        ])
                    })
                    .collect(),
            ),
        ),
        ("energy", Json::n(sol.energy)),
        // What the answer is WORTH, in the modeller's own units and the direction they wrote it.
        // "energy" above is the compiled Ising energy with every penalty and the constant folded
        // in: it compares two answers to one model and nothing else, and it moves when the penalty
        // does. Null when no objective was written, when both senses were used and there is no
        // single direction to report, or when a variable did not decode.
        ("objective", sol.objective.map_or(Json::Null, Json::n)),
        // Only "branch" can set this. Read it WITH "feasible": branch proves a statement about the
        // compiled energy, and it becomes a statement about the caller's model exactly when the
        // answer is also feasible, because a feasible assignment pays no penalty and its compiled
        // energy is the objective plus a constant.
        ("proved_optimal", Json::Bool(sol.proved_optimal)),
        // What the traded-away preferences cost, separated from the objective on purpose: telling
        // those apart is the whole point of saying a constraint is soft.
        ("soft_cost", Json::n(sol.soft_cost())),
        // What the compiler knows is wrong with the model and cannot fix. Empty is normal; a
        // non-empty one means a value that reads back fine may have come from a codeword the
        // penalty never excluded.
        (
            "caveats",
            Json::Arr(compiled.caveats.iter().map(|c| Json::s(c)).collect()),
        ),
        ("spins", Json::n(compiled.spins() as f64)),
        // Spins the higher-order lowering added. Non-zero means the answer solves a model with more
        // spins than the variables required, and that the guarantee is about optima not sampling.
        ("ancillas", Json::n(compiled.ancillas as f64)),
        ("penalty", Json::n(penalty)),
        ("tries", Json::n(tries as f64)),
        ("wall_seconds", Json::n(wall)),
        ("ftp", Json::Str(compiled.program.to_ftp())),
        // The same program as an ommx.v1.Instance, base64 because JSON has no bytes. OMMX is the
        // interchange format this corner of the field converged on, so this is what makes a
        // ferrotherm answer readable by everyone else's tooling.
        ("ommx_b64", Json::Str(b64(&ferrotherm::ommx::export(&compiled.graph).bytes))),
        // ferrotherm_energy(s) == ommx_objective(x) + this. Dropping it would hand back an instance
        // with the same optimum and the wrong value.
        ("ommx_constant", Json::n(ferrotherm::ommx::export(&compiled.graph).constant)),
        (
            "note",
            Json::s(
                "`feasible: false` means the answer breaks something you asked for: either a \
                 variable in \"did_not_decode\" whose encoding was violated, or a HARD constraint \
                 in \"violated\" that the objective outbid. A penalty makes a constraint expensive, \
                 not impossible. Raise \"penalty\" or lower the objective weights and try again; \
                 if it stays infeasible at a large penalty, the problem itself may have no answer. \
                 Entries with \"hard\": false are preferences you priced with \"soft\", and the \
                 solver trading one away leaves the answer feasible -- \"soft_cost\" totals what \
                 those trades cost, at weight x amount SQUARED. \
                 \"objective\" is what the answer is worth in YOUR units and the direction you wrote it; \"energy\" is the compiled Ising energy with every penalty and the constant folded in, which compares two answers to one model and nothing else. The \"ftp\" field is the compiled program and runs unchanged on any backend.",
            ),
        ),
    ]))
}

/// Route a named operation. Shared by both transports.
/// Minimise a HIGHER-ORDER model natively, with no ancillas.
///
/// Every other operation here is pairwise, or becomes pairwise. `solve` will take a term over three
/// or more variables and lower it through Rosenberg's reduction -- one ancilla per substituted pair
/// plus a penalty -- and report the ancillas it spent. That is the right pass when the target is
/// pairwise hardware and the wrong one when the target is a CPU, and the difference is measured
/// rather than argued: `examples/hubo_vs_reduction` gives the reduced arm its best beta ladder and
/// 1024x the budget, and on 60 three-body terms over 40 spins it reaches -34.00 where this path
/// reaches -48.12 at 1x. The mechanism is the penalty, ~1300 against term weights of 1, which makes
/// the landscape rigid rather than merely larger.
///
/// Terms are `{"vars": [i, j, k, ...], "weight": w}` over 0-based spin indices, any arity.
pub fn hubo(req: &Json) -> Result<Json, String> {
    let n = value_of(req, "spins", 0, "hubo")?;
    if n < 1 {
        return Err("\"spins\" must be at least 1: a model with no variables can hold no term".into());
    }
    let terms = req
        .get("terms")
        .and_then(|x| x.as_arr())
        .ok_or("missing \"terms\": an array of {\"vars\": [i, j, k], \"weight\": w}")?;
    if terms.is_empty() {
        return Err("\"terms\" is empty; a model with no terms has nothing to minimise".into());
    }

    let mut h = ferrotherm::hubo::Hubo::new(n as usize);
    for (i, term) in terms.iter().enumerate() {
        let vs = term
            .get("vars")
            .and_then(|x| x.as_arr())
            .ok_or_else(|| format!("terms[{i}] needs a \"vars\" array of spin indices"))?;
        let mut vars = Vec::with_capacity(vs.len());
        for (k, v) in vs.iter().enumerate() {
            let x = v
                .as_i64()
                .ok_or_else(|| format!("terms[{i}].vars[{k}] must be a whole number"))?;
            if x < 0 || x >= n {
                return Err(format!(
                    "terms[{i}].vars[{k}] is {x}, and this model has {n} spins numbered 0..{}",
                    n - 1
                ));
            }
            vars.push(x as usize);
        }
        let w = term
            .get("weight")
            .and_then(|x| x.as_f64())
            .ok_or_else(|| format!("terms[{i}] needs a numeric \"weight\""))?;
        // The library's refusal is more specific than anything phrased here -- it names the
        // repeated variable and how many times it appeared -- so it is passed through.
        h.add(&vars, w).map_err(|e| format!("terms[{i}]: {e}"))?;
    }

    let d = ferrotherm::hubo::Params::default();
    let p = ferrotherm::hubo::Params {
        beta_min: opt_f64(req, "beta_min", d.beta_min),
        beta_max: opt_f64(req, "beta_max", d.beta_max),
        stages: opt_usize(req, "stages", d.stages),
        sweeps_per_stage: opt_usize(req, "sweeps_per_stage", d.sweeps_per_stage),
    };
    if !(p.beta_max > p.beta_min) || !p.beta_min.is_finite() || !p.beta_max.is_finite() {
        return Err(format!(
            "beta_max must exceed beta_min and both must be real; got {} and {}",
            p.beta_min, p.beta_max
        ));
    }
    let seed = value_of(req, "seed", 1, "hubo")? as u64;
    let out = ferrotherm::hubo::anneal(&h, &p, seed);

    Ok(Json::obj(vec![
        ("energy", Json::n(out.energy)),
        ("state", Json::Arr(out.state.iter().map(|&s| Json::n(s as f64)).collect())),
        ("spins", Json::n(h.len() as f64)),
        ("terms", Json::n(h.terms() as f64)),
        ("max_arity", Json::n(h.max_arity() as f64)),
        ("ancillas_avoided", Json::n(h.ancillas_avoided() as f64)),
        ("proposals", Json::n(out.proposals as f64)),
        ("accepted", Json::n(out.accepted as f64)),
        (
            "note",
            Json::s(
                "Solved natively: no ancillas, and no penalty weight to get right. \
                 \"ancillas_avoided\" is an UPPER BOUND on what a pairwise reduction would have \
                 spent, not the cost -- the reduction shares one ancilla across every term \
                 containing the same pair, so on three terms sharing one it spends one where this \
                 reports three. Use \"solve\" instead when the target is pairwise hardware, which \
                 is what the reduction is for.",
            ),
        ),
    ]))
}

/// Fit a Boltzmann machine to data, and hand back the fitted model as a graph.
///
/// EVERY OTHER OPERATION HERE CONSUMES A MODEL. This one produces one, and it is the operation that
/// makes this server a computing paradigm rather than a solver behind HTTP: the argument for this
/// class of hardware is that it samples Boltzmann distributions cheaply, and the distributions
/// anyone actually wants are FITTED.
///
/// The reply carries the fitted weights in the same `graph` shape every other operation TAKES, so
/// an agent can fit here and then sample, anneal, bound or certify the result with no export step
/// and no second format. That round trip is the whole point of returning a graph rather than an
/// opaque handle.
pub fn fit(req: &Json) -> Result<Json, String> {
    let visible = value_of(req, "visible", 0, "fit")?;
    if visible < 1 {
        return Err("\"visible\" must be at least 1: a machine with no visible units has nothing \
                    to clamp data onto"
            .into());
    }
    let visible = visible as usize;

    let hidden: Vec<usize> = match req.get("hidden") {
        Some(h) => {
            let arr = h
                .as_arr()
                .ok_or("\"hidden\" is an array of layer widths, outermost last: [12] is an RBM, \
                        [6, 6] is two layers")?;
            let mut out = Vec::with_capacity(arr.len());
            for (i, w) in arr.iter().enumerate() {
                let w = w
                    .as_i64()
                    .ok_or_else(|| format!("hidden[{i}] must be a whole number"))?;
                if w < 1 {
                    return Err(format!("hidden[{i}] is {w}, and a layer needs at least one unit"));
                }
                out.push(w as usize);
            }
            out
        }
        None => return Err("missing \"hidden\": [12] for a restricted Boltzmann machine of twelve \
                            hidden units, [6, 6] for a two-layer deep one".into()),
    };
    if hidden.is_empty() {
        return Err("\"hidden\" is empty; a machine with no latent units can only learn the \
                    independent statistics of each visible unit"
            .into());
    }

    let rows = dataset_from(req, visible)?;
    let structure = ferrotherm::ebm::dbm(visible, &hidden);
    let data = ferrotherm::ebm::Dataset { visible, rows };

    let d = ferrotherm::ebm::Params::default();
    let p = ferrotherm::ebm::Params {
        epochs: opt_usize(req, "epochs", d.epochs),
        k: opt_usize(req, "k", d.k),
        positive_sweeps: opt_usize(req, "positive_sweeps", d.positive_sweeps),
        learning_rate: opt_f64(req, "learning_rate", d.learning_rate),
        batch: opt_usize(req, "batch", d.batch),
    };
    if !(p.learning_rate > 0.0) || !p.learning_rate.is_finite() {
        return Err(format!(
            "\"learning_rate\" must be a positive real; got {}",
            p.learning_rate
        ));
    }
    let seed = value_of(req, "seed", 1, "fit")? as u64;

    // The untrained score is not measured, it is DERIVED: every weight starts at zero, so the model
    // is uniform and can only score -visible*ln2. Reporting it beside the fit is what makes the
    // fitted number readable -- a log-likelihood alone tells a caller nothing about whether it is
    // good, and a caller that cannot tell will either trust a bad fit or discard a good one.
    let before = ferrotherm::ebm::exact_log_likelihood(&structure, &data).ok();
    let out = ferrotherm::ebm::train(&structure, &data, &p, seed).map_err(|e| e.to_string())?;
    let g = &out.graph;

    let floor = -(visible as f64) * core::f64::consts::LN_2;
    let ceiling = -(data.rows.len() as f64).ln();
    let learned = out.log_likelihood.map(|v| (v - floor) / (ceiling - floor) * 100.0);

    let mut couplings = Vec::new();
    for i in 0..g.n {
        for k in g.offset[i]..g.offset[i + 1] {
            let j = g.nbr[k] as usize;
            if j > i {
                couplings.push(Json::Arr(vec![
                    Json::n(i as f64),
                    Json::n(j as f64),
                    Json::n(g.w[k]),
                ]));
            }
        }
    }
    let biases: Vec<Json> = g
        .h
        .iter()
        .enumerate()
        .filter(|(_, &b)| b != 0.0)
        .map(|(i, &b)| Json::Arr(vec![Json::n(i as f64), Json::n(b)]))
        .collect();

    let num = |v: Option<f64>| v.map(Json::n).unwrap_or(Json::Null);
    Ok(Json::obj(vec![
        ("log_likelihood", num(out.log_likelihood)),
        ("log_likelihood_untrained", num(before)),
        ("learned_percent", num(learned)),
        (
            "scale",
            Json::obj(vec![
                ("learned_nothing", Json::n(floor)),
                ("learned_everything", Json::n(ceiling)),
            ]),
        ),
        ("visible", Json::n(visible as f64)),
        ("hidden", Json::Arr(hidden.iter().map(|&w| Json::n(w as f64)).collect())),
        ("spins", Json::n(g.n as f64)),
        ("edges", Json::n(g.n_edges as f64)),
        ("rows", Json::n(data.rows.len() as f64)),
        (
            "graph",
            Json::obj(vec![
                ("n", Json::n(g.n as f64)),
                ("couplings", Json::Arr(couplings)),
                ("biases", Json::Arr(biases)),
            ]),
        ),
        (
            "note",
            Json::s(
                "\"graph\" is the fitted model in the shape every other operation TAKES: pass it \
                 to sample, anneal, bound or verify with no export step. The likelihood is EXACT, \
                 by enumeration, and is null above 22 spins where enumerating is refused rather \
                 than replaced by a cheaper estimate -- an ELBO, a reconstruction error or a \
                 pseudo-likelihood is worst exactly where sampling is worst, so comparing machines \
                 on one reads the proxy's failure and calls it expressivity. \"learned_percent\" \
                 places the fit between two ends that need no calibration: an untrained model is \
                 uniform over 2^visible images, and a perfect one is uniform over the rows you gave.",
            ),
        ),
    ]))
}

/// The dataset a fit request names, either by benchmark or written out.
fn dataset_from(req: &Json, visible: usize) -> Result<Vec<Vec<i8>>, String> {
    let d = req.get("data").ok_or(
        "missing \"data\": either rows of -1 and +1, or the name \"bars-and-stripes-N\"",
    )?;
    if let Some(name) = d.as_str() {
        let side = name
            .strip_prefix("bars-and-stripes-")
            .and_then(|s| s.parse::<usize>().ok())
            .ok_or_else(|| {
                format!(
                    "{name:?} is not a dataset this server knows; the only named one is \
                     \"bars-and-stripes-N\", and otherwise give rows of -1 and +1"
                )
            })?;
        if !(1..=8).contains(&side) {
            return Err(format!("\"bars-and-stripes-{side}\" needs N between 1 and 8"));
        }
        let ds = ferrotherm::ebm::bars_and_stripes(side);
        if ds.visible != visible {
            return Err(format!(
                "\"bars-and-stripes-{side}\" has {} entries per row and the machine has {visible} \
                 visible units; they have to match, because a row is clamped onto them",
                ds.visible
            ));
        }
        return Ok(ds.rows);
    }
    let arr = d.as_arr().ok_or(
        "\"data\" is either an array of rows or the name \"bars-and-stripes-N\"",
    )?;
    if arr.is_empty() {
        return Err("\"data\" is empty; there is nothing to fit".into());
    }
    let mut rows = Vec::with_capacity(arr.len());
    for (i, row) in arr.iter().enumerate() {
        let cells = row
            .as_arr()
            .ok_or_else(|| format!("data[{i}] must be an array of -1 and +1"))?;
        if cells.len() != visible {
            return Err(format!(
                "data[{i}] has {} entries and the machine has {visible} visible units",
                cells.len()
            ));
        }
        let mut out = Vec::with_capacity(visible);
        for (j, c) in cells.iter().enumerate() {
            match c.as_f64() {
                Some(1.0) => out.push(1i8),
                Some(-1.0) => out.push(-1i8),
                _ => {
                    return Err(format!(
                        "data[{i}][{j}] must be -1 or +1; a spin has no other values"
                    ))
                }
            }
        }
        rows.push(out);
    }
    Ok(rows)
}

/// A starting state from the request, in the shape `branch`'s `incumbent` already uses.
///
/// `None` when absent. A wrong length is REFUSED here rather than ignored: over an HTTP boundary a
/// caller cannot see that their state was dropped, and a search that silently started from noise
/// would return a worse answer with nothing to say why. The library ignores it because a Rust
/// caller can read the field back; a request cannot.
fn start_from(req: &Json, n: usize) -> Result<Option<Vec<i8>>, String> {
    let Some(sv) = req.get("incumbent").and_then(|s| s.as_arr()) else { return Ok(None) };
    if sv.len() != n {
        return Err(format!(
            "\"incumbent\" has {} entries but the graph has {n} nodes",
            sv.len()
        ));
    }
    Ok(Some(
        sv.iter().map(|x| if x.as_f64() == Some(1.0) { 1i8 } else { -1 }).collect(),
    ))
}

pub fn dispatch(op: &str, req: &Json) -> Result<Json, String> {
    match op {
        "sample" => sample(req),
        "anneal" => anneal(req),
        "energy" => energy(req),
        "verify" => verify(req),
        "bound" => bound(req),
        "optimize" => optimize(req),
        "exact_planar" => exact_planar(req),
        "toroidal_bound" => toroidal_bound(req),
        "solve" => solve(req),
        "hubo" => hubo(req),
        "fit" => fit(req),
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

    /// THE ROUND TRIP IS THE FEATURE. `fit` returns a graph in the shape every other operation
    /// takes, and the only way to know that is still true is to take one operation's output and
    /// hand it to the next ones unmodified. A reply that merely CONTAINS a "graph" key satisfies
    /// any schema check and can still be unusable.
    #[test]
    fn a_fitted_model_is_accepted_by_the_operations_that_consume_models() {
        let req = crate::json::parse(
            r#"{"visible":9,"hidden":[12],"data":"bars-and-stripes-3","epochs":200,"k":10,"seed":3}"#,
        )
        .unwrap();
        let out = fit(&req).expect("a 21-spin machine on 14 rows must fit");

        // Both ends of the scale are DERIVED, not measured: every weight starts at zero, so an
        // untrained machine is uniform over 2^9 images and can only score -9 ln 2.
        let untrained = out.get("log_likelihood_untrained").and_then(|v| v.as_f64()).unwrap();
        assert!((untrained + 9.0 * core::f64::consts::LN_2).abs() < 1e-12, "{untrained}");
        let learned = out.get("learned_percent").and_then(|v| v.as_f64()).unwrap();
        assert!(learned > 85.0, "a wide machine on this data reaches the nineties: {learned}");

        let g = out.get("graph").expect("the fit must return its model");
        assert_eq!(g.get("n").and_then(|v| v.as_f64()), Some(21.0));
        assert_eq!(g.get("couplings").and_then(|v| v.as_arr()).map(|a| a.len()), Some(108));

        // Now the round trip, through the same `graph_from` every consuming operation uses. A
        // fitted model is DENSE compared with the lattices these operations usually see, so this
        // also checks the marshalling at a shape the rest of the suite never produces.
        let with_graph = |extra: &str| {
            crate::json::parse(&format!(
                "{{\"graph\":{}{}}}",
                crate::json::write(g),
                extra
            ))
            .unwrap()
        };
        let annealed = anneal(&with_graph(r#","seed":1"#)).expect("anneal takes a fitted model");
        let e = annealed.get("best_energy").and_then(|v| v.as_f64()).unwrap();

        let bounded = bound(&with_graph("")).expect("bound takes a fitted model");
        let b = bounded.get("best").and_then(|v| v.as_f64()).unwrap();
        // One-sided and the only direction that can be asserted: a sound bound never exceeds an
        // energy actually attained.
        assert!(b <= e + 1e-9, "bound {b} must not exceed an attained energy {e}");

        let sampled = sample(&with_graph(r#","beta":1.0,"sweeps":200,"seed":2"#))
            .expect("sample takes a fitted model");
        assert!(sampled.get("energy").and_then(|v| v.as_f64()).unwrap().is_finite());
    }

    /// A fit request that cannot be honoured says which part, in the caller's own terms.
    #[test]
    fn a_bad_fit_request_names_the_part_that_is_wrong() {
        let bad = |body: &str| fit(&crate::json::parse(body).unwrap()).unwrap_err();

        let e = bad(r#"{"visible":9,"data":"bars-and-stripes-3"}"#);
        assert!(e.contains("hidden"), "{e}");
        let e = bad(r#"{"visible":9,"hidden":[12]}"#);
        assert!(e.contains("data"), "{e}");
        // The commonest real mistake: a machine whose visible width does not match the data's.
        let e = bad(r#"{"visible":4,"hidden":[8],"data":"bars-and-stripes-3"}"#);
        assert!(e.contains("clamped"), "it says WHY they must match: {e}");
        let e = bad(r#"{"visible":9,"hidden":[12],"data":"mnist"}"#);
        assert!(e.contains("bars-and-stripes"), "{e}");
        let e = bad(r#"{"visible":2,"hidden":[2],"data":[[1,0]]}"#);
        assert!(e.contains("-1 or +1"), "{e}");
        let e = bad(r#"{"visible":2,"hidden":[0],"data":[[1,-1]]}"#);
        assert!(e.contains("at least one unit"), "{e}");

        // Above the enumeration ceiling the fit still runs; only its SCORE is refused, and it is
        // refused by returning null rather than by substituting something cheaper.
        let big = crate::json::parse(
            r#"{"visible":16,"hidden":[12],"data":[[1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1]],"epochs":5}"#,
        )
        .unwrap();
        let out = fit(&big).expect("a 28-spin machine still fits");
        assert!(matches!(out.get("log_likelihood"), Some(Json::Null)));
        assert!(out.get("graph").is_some(), "the model is real even when its score is not taken");
    }

    #[test]
    fn a_quadratic_model_is_refused_by_coupling_count_not_node_count() {
        // 46 bytes: {"variables":[{"name":"x","values":1000}],"tries":1}. It compiles to 1000
        // spins -- comfortably under MAX_NODES -- and one sweep, so no node-update bound fires
        // either. But a one-hot over k values carries k(k-1)/2 couplings: 499,500 of them, which
        // took 6.7 s and returned a 17 MB reply. Every existing ceiling measured a dimension that
        // was not the one growing.
        let req = crate::json::parse(r#"{"variables":[{"name":"x","values":1000}],"tries":1}"#).unwrap();
        let Err(e) = solve(&req) else { panic!("499,500 couplings from 46 bytes must be refused") };
        assert!(e.contains("couplings"), "names the dimension that grew: {e}");
        assert!(e.contains("DomainWall"), "and the encoding that is linear: {e}");

        // Well under the ceiling, and still served.
        let ok = crate::json::parse(r#"{"variables":[{"name":"x","values":60}],"tries":1}"#).unwrap();
        assert!(solve(&ok).is_ok(), "1,770 couplings is an ordinary model");
    }

    #[test]
    fn an_enormous_stage_count_saturates_rather_than_wrapping() {
        // `(stages * per) as u64` multiplies in USIZE first, so it wrapped BEFORE the cast: a
        // stage count of 2^63 produced a small budget, sailed past the node-update ceiling, and
        // aborted the server in raw_vec with a capacity overflow -- an empty reply, no 400, and
        // the process gone.
        let req = crate::json::parse(
            r#"{"graph":{"builtin":"ring","n":8},"stages":9223372036854775808,"sweeps_per_stage":2}"#,
        )
        .unwrap();
        // Either bounded and served, or refused -- never a panic, and never a wrapped budget.
        match anneal(&req) {
            Ok(_) => {}
            Err(e) => assert!(!e.is_empty(), "a refusal must say something"),
        }
    }
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

    /// A bound is a bound, over HTTP as much as in Rust — and the gap is what the tool is for.
    #[test]
    fn bounds_never_exceed_an_optimum_the_same_server_can_prove() {
        // A frustrated ring: n bonds, all but one satisfiable, and small enough to prove.
        let mut edges = Vec::new();
        for i in 0..12 {
            let w = if i == 0 { -1.0 } else { 1.0 };
            edges.push(format!("[{i},{},{w}]", (i + 1) % 12));
        }
        let g = format!(r#"{{"n":12,"couplings":[{}]}}"#, edges.join(","));

        let proof = dispatch("optimize", &parse(&format!(r#"{{"graph":{g},"method":"branch"}}"#)).unwrap()).unwrap();
        assert_eq!(proof.get("proved_optimal").and_then(|v| v.as_bool()), Some(true));
        let truth = proof.get("best_energy").unwrap().as_f64().unwrap();
        assert!((truth - -10.0).abs() < 1e-9, "a frustrated 12-ring bottoms out at -10, got {truth}");

        let state = crate::json::write(proof.get("state").unwrap());
        let b = dispatch("bound", &parse(&format!(r#"{{"graph":{g},"state":{state}}}"#)).unwrap()).unwrap();
        for k in ["decoupled", "forest", "odd_cycle"] {
            let v = b.get(k).unwrap().as_f64().unwrap();
            assert!(v <= truth + 1e-9, "{k} bound {v} exceeds the proved minimum {truth}");
        }
        // `sdp` is a number when the certificate re-verified and a REFUSAL STRING when it did not.
        // Either is fine; a wrong number is not.
        if let Some(v) = b.get("sdp").and_then(|v| v.as_f64()) {
            assert!(v <= truth + 1e-9, "sdp bound {v} exceeds the proved minimum {truth}");
        }
        let best = b.get("best").unwrap().as_f64().unwrap();
        assert!(best <= truth + 1e-9);
        assert_eq!(b.get("energy").and_then(|v| v.as_f64()), Some(truth));
        // The state IS the optimum, so the gap is exactly how loose the best bound is.
        assert!((b.get("gap").unwrap().as_f64().unwrap() - (truth - best)).abs() < 1e-9);
        // On a ring with no fields, `forest` cannot beat `decoupled`: a tree is never frustrated.
        assert!(
            (b.get("forest").unwrap().as_f64().unwrap()
                - b.get("decoupled").unwrap().as_f64().unwrap())
            .abs()
                < 1e-9
        );
    }

    /// The one operation here that returns an OPTIMUM, and the four refusals that are four
    /// different instructions.
    #[test]
    fn exact_planar_returns_a_maximum_and_names_why_it_will_not() {
        // A 4x4 antiferromagnetic grid: bipartite, so every one of its 24 edges is cut.
        let mut e = Vec::new();
        for y in 0..4usize {
            for x in 0..4usize {
                let i = y * 4 + x;
                if x + 1 < 4 {
                    e.push(format!("[{i},{},-1]", i + 1));
                }
                if y + 1 < 4 {
                    e.push(format!("[{i},{},-1]", i + 4));
                }
            }
        }
        let g = format!(r#"{{"n":16,"couplings":[{}]}}"#, e.join(","));
        let r = dispatch("exact_planar", &parse(&format!(r#"{{"graph":{g}}}"#)).unwrap()).unwrap();
        assert_eq!(r.get("cut").and_then(|v| v.as_f64()), Some(24.0));
        assert_eq!(r.get("energy").and_then(|v| v.as_f64()), Some(-24.0));
        assert_eq!(r.get("exact").and_then(|v| v.as_bool()), Some(true));
        assert_eq!(r.get("faces").and_then(|v| v.as_f64()), Some(10.0));

        // And no search can beat it, which is the check that makes "exact" mean something.
        let o = dispatch("optimize", &parse(&format!(
            r#"{{"graph":{g},"method":"breakout","iterations":50000}}"#)).unwrap()).unwrap();
        assert!(o.get("best_energy").unwrap().as_f64().unwrap() >= -24.0 - 1e-9);

        // A periodic lattice is a TORUS. Refused, and the reply says which of the four it is.
        let torus = r#"{"builtin":"lattice2d","l":4,"j":1.0}"#;
        let err = dispatch("exact_planar", &parse(&format!(r#"{{"graph":{torus}}}"#)).unwrap())
            .unwrap_err();
        assert!(err.contains("not planar"), "{err}");

        // A field makes it a different problem, not a harder one.
        let fielded = format!(r#"{{"n":16,"couplings":[{}],"biases":[[0,0.5]]}}"#, e.join(","));
        let err = dispatch("exact_planar", &parse(&format!(r#"{{"graph":{fielded}}}"#)).unwrap())
            .unwrap_err();
        assert!(err.contains("field"), "{err}");
    }

    /// The other end of the bracket, over HTTP.
    #[test]
    fn the_toroidal_bound_bounds_and_declines_what_is_not_a_torus() {
        let torus = r#"{"builtin":"lattice2d","l":6,"j":-1.0}"#;
        let r = dispatch("toroidal_bound", &parse(&format!(r#"{{"graph":{torus}}}"#)).unwrap())
            .unwrap();
        // A 6x6 periodic lattice is bipartite: all 72 edges cut, and the bound is achieved.
        assert_eq!(r.get("upper_bound").and_then(|v| v.as_f64()), Some(72.0));
        assert_eq!(r.get("attained").and_then(|v| v.as_bool()), Some(true));
        assert_eq!(r.get("genus").and_then(|v| v.as_f64()), Some(1.0));

        // The planar solver declines the same graph, which is the distinction being drawn.
        assert!(dispatch("exact_planar", &parse(&format!(r#"{{"graph":{torus}}}"#)).unwrap())
            .unwrap_err()
            .contains("not planar"));

        // And this one declines an open grid, saying it is a statement about the graph.
        let mut e = Vec::new();
        for y in 0..4usize {
            for x in 0..4usize {
                let i = y * 4 + x;
                if x + 1 < 4 {
                    e.push(format!("[{i},{},-1]", i + 1));
                }
                if y + 1 < 4 {
                    e.push(format!("[{i},{},-1]", i + 4));
                }
            }
        }
        let open = format!(r#"{{"n":16,"couplings":[{}]}}"#, e.join(","));
        let err = dispatch("toroidal_bound", &parse(&format!(r#"{{"graph":{open}}}"#)).unwrap())
            .unwrap_err();
        assert!(err.contains("not a toroidal grid"), "{err}");
    }

    /// The three algorithms the toolchain survey named as missing, over HTTP, each carrying the
    /// caveat that makes it honest.
    #[test]
    fn the_closed_gaps_report_their_own_caveats_over_http() {
        // A 6x6 ANTIferromagnet is bipartite and inside the GW hypothesis: 72 cuttable edges.
        let anti = r#"{"builtin":"lattice2d","l":6,"j":-1.0}"#;
        let r = dispatch("optimize", &parse(&format!(
            r#"{{"graph":{anti},"method":"goemans_williamson","hyperplanes":64}}"#)).unwrap()).unwrap();
        assert_eq!(r.get("guaranteed").and_then(|v| v.as_bool()), Some(true));
        assert_eq!(r.get("cut").and_then(|v| v.as_f64()), Some(72.0));

        // A ferromagnet is OUTSIDE it, and the reply must say so rather than claim a ratio.
        let ferro = r#"{"builtin":"lattice2d","l":6,"j":1.0}"#;
        let f = dispatch("optimize", &parse(&format!(
            r#"{{"graph":{ferro},"method":"goemans_williamson"}}"#)).unwrap()).unwrap();
        assert_eq!(f.get("guaranteed").and_then(|v| v.as_bool()), Some(false));
        assert!(f.get("guarantee_note").unwrap().as_str().unwrap().contains("no ratio"));

        // Cluster moves fire and find the ferromagnetic ground state.
        let c = dispatch("optimize", &parse(&format!(
            r#"{{"graph":{ferro},"method":"cluster","rungs":8,"rounds":200}}"#)).unwrap()).unwrap();
        assert_eq!(c.get("best_energy").and_then(|v| v.as_f64()), Some(-72.0));
        assert!(c.get("cluster_moves").unwrap().as_f64().unwrap() > 0.0);

        // Simulated quantum annealing, and the note that says what it is not.
        let q = dispatch("optimize", &parse(&format!(
            r#"{{"graph":{ferro},"method":"quantum","trotter":4,"steps":200}}"#)).unwrap()).unwrap();
        assert_eq!(q.get("best_energy").and_then(|v| v.as_f64()), Some(-72.0));
        assert!(q.get("note").unwrap().as_str().unwrap().contains("no quantum claim"));
        // One slice is the CONTROL and says so, rather than being a silent degenerate case.
        let one = dispatch("optimize", &parse(&format!(
            r#"{{"graph":{ferro},"method":"quantum","trotter":1}}"#)).unwrap()).unwrap();
        assert!(one.get("note").unwrap().as_str().unwrap().contains("CLASSICAL"));
        assert_eq!(one.get("max_j_perp").and_then(|v| v.as_f64()), Some(0.0));

        // A field breaks the isoenergetic argument, and the reason is the reply.
        let mut e = Vec::new();
        for i in 0..8usize {
            e.push(format!("[{i},{},1]", (i + 1) % 8));
        }
        let fielded = format!(r#"{{"n":8,"couplings":[{}],"biases":[[3,0.5]]}}"#, e.join(","));
        let err = dispatch("optimize", &parse(&format!(
            r#"{{"graph":{fielded},"method":"cluster"}}"#)).unwrap()).unwrap_err();
        assert!(err.contains("isoenergetic"), "{err}");
    }

    /// Each method reports the thing only it can report, and a bad name is refused rather than
    /// silently defaulted.
    #[test]
    fn optimize_carries_each_method_own_diagnostic() {
        let g = r#"{"builtin":"ring","n":16,"j":1.0,"h":0.0}"#;

        let tabu = dispatch("optimize", &parse(&format!(
            r#"{{"graph":{g},"method":"tabu","iterations":2000}}"#)).unwrap()).unwrap();
        assert_eq!(tabu.get("iterations_run").and_then(|v| v.as_f64()), Some(2000.0));

        let pa = dispatch("optimize", &parse(&format!(
            r#"{{"graph":{g},"method":"population","population":128,"stages":20}}"#)).unwrap()).unwrap();
        let rho = pa.get("rho").unwrap().as_f64().unwrap();
        assert!((1.0..=128.0).contains(&rho), "rho {rho} outside [1, population]");
        assert!(pa.get("rho_reading").and_then(|v| v.as_str()).is_some());
        // Z(0) = 2^n and Z never falls as beta rises, so ln Z is at least n ln 2.
        let ln_z = pa.get("ln_z").unwrap().as_f64().unwrap();
        assert!(ln_z >= 16.0 * std::f64::consts::LN_2 - 1e-9, "ln_z {ln_z}");

        let br = dispatch("optimize", &parse(&format!(
            r#"{{"graph":{g},"method":"branch","max_nodes":5}}"#)).unwrap()).unwrap();
        assert_eq!(br.get("proved_optimal").and_then(|v| v.as_bool()), Some(false));
        assert_eq!(br.get("hit_limit").and_then(|v| v.as_bool()), Some(true));

        let bl = dispatch("optimize", &parse(&format!(
            r#"{{"graph":{g},"method":"breakout","iterations":20000}}"#)).unwrap()).unwrap();
        assert_eq!(bl.get("iterations_run").and_then(|v| v.as_f64()), Some(20000.0));
        // The claim BLS makes is about what happens BETWEEN local optima. A run with one descent
        // spent 20,000 flips inside a single basin and is not the algorithm.
        assert!(bl.get("descents").unwrap().as_f64().unwrap() > 1.0, "{:?}", bl.get("descents"));
        let mix = bl.get("perturbations").expect("the adaptive mix is the algorithm");
        let total: f64 = ["directed_one", "directed_two", "random"]
            .iter()
            .map(|k| mix.get(k).and_then(|v| v.as_f64()).unwrap_or(-1.0))
            .sum();
        assert!(total > 0.0, "no perturbation of any kind fired: {mix:?}");

        let e = dispatch("optimize", &parse(&format!(r#"{{"graph":{g},"method":"anneal"}}"#)).unwrap());
        assert!(e.unwrap_err().contains("unknown method"), "a typo must not silently pick a default");
    }

    /// Every solver returns the energy OF THE STATE IT RETURNS, recomputed rather than carried.
    #[test]
    fn the_energy_over_http_belongs_to_the_state_over_http() {
        let g = r#"{"builtin":"lattice2d","l":6,"j":1.0}"#;
        for method in ["tabu", "breakout", "population", "branch"] {
            let r = dispatch("optimize", &parse(&format!(
                r#"{{"graph":{g},"method":"{method}","iterations":2000,"population":32,"stages":10,"max_nodes":50000}}"#
            )).unwrap()).unwrap();
            let reported = r.get("best_energy").unwrap().as_f64().unwrap();
            let state = crate::json::write(r.get("state").unwrap());
            let scored = dispatch("energy", &parse(&format!(r#"{{"graph":{g},"state":{state}}}"#)).unwrap())
                .unwrap()
                .get("energy")
                .unwrap()
                .as_f64()
                .unwrap();
            assert!(
                (reported - scored).abs() < 1e-9,
                "{method} reported {reported}, its own state scores {scored}"
            );
        }
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
        let over = r#"{"graph":{"builtin":"lattice2d","l":64},"sweeps":1,"draws":100000,"thin":100000}"#.to_string();
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

    #[test]
    fn a_soft_constraint_is_traded_and_priced_while_a_hard_one_is_not() {
        // The same model twice, differing only in whether the constraint is a rule or a price.
        // What changes is not whether the solver CAN break it -- a penalty was always breakable --
        // but what the answer means when it does, and the reply has to say which happened.
        let body = |c: &str| {
            format!(
                r#"{{"variables":[{{"name":"a","values":2}},{{"name":"b","values":2}}],
                    "constraints":[{{"type":"not_equal","a":"a","b":"b"{c}}}],
                    "objective":{{"maximize":true,"terms":[
                       {{"var":"a","value":0,"weight":5}},{{"var":"b","value":0,"weight":5}}]}},
                    "tries":24}}"#
            )
        };

        let soft = dispatch("solve", &crate::json::parse(&body(r#","soft":1"#)).unwrap()).unwrap();
        assert_eq!(soft.get("feasible").unwrap().as_bool(), Some(true), "{soft:?}");
        let broke = soft.get("violated").unwrap().as_arr().unwrap();
        assert_eq!(broke.len(), 1, "the cheap preference should have been traded: {soft:?}");
        assert_eq!(broke[0].get("hard").unwrap().as_bool(), Some(false));
        assert_eq!(broke[0].get("cost").unwrap().as_f64(), Some(1.0));
        assert_eq!(soft.get("soft_cost").unwrap().as_f64(), Some(1.0));

        // The identical constraint as a rule: the solver keeps it and gives up the objective.
        let hard = dispatch("solve", &crate::json::parse(&body("")).unwrap()).unwrap();
        assert_eq!(hard.get("feasible").unwrap().as_bool(), Some(true), "{hard:?}");
        assert!(hard.get("violated").unwrap().as_arr().unwrap().is_empty(), "{hard:?}");
        assert_eq!(hard.get("soft_cost").unwrap().as_f64(), Some(0.0));

        // And priced above the objective, the preference is kept rather than traded. Same field,
        // opposite outcome: this is the knob doing what it says.
        let dear = dispatch("solve", &crate::json::parse(&body(r#","soft":50"#)).unwrap()).unwrap();
        assert!(dear.get("violated").unwrap().as_arr().unwrap().is_empty(), "{dear:?}");
        assert_eq!(dear.get("soft_cost").unwrap().as_f64(), Some(0.0));
    }

    #[test]
    fn a_soft_price_is_squared_over_the_wire() {
        let r = dispatch(
            "solve",
            &crate::json::parse(
                r#"{"variables":[{"name":"v0","values":2},{"name":"v1","values":2},
                                 {"name":"v2","values":2},{"name":"v3","values":2}],
                    "constraints":[{"type":"at_most","k":1,"soft":1,"of":[
                       {"var":"v0","value":1},{"var":"v1","value":1},
                       {"var":"v2","value":1},{"var":"v3","value":1}]}],
                    "objective":{"maximize":true,"terms":[
                       {"var":"v0","value":1,"weight":20},{"var":"v1","value":1,"weight":20},
                       {"var":"v2","value":1,"weight":20},{"var":"v3","value":1,"weight":20}]},
                    "tries":24}"#,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(r.get("feasible").unwrap().as_bool(), Some(true), "{r:?}");
        // All four held against a cap of one, so it is over by three -- and three squared is nine,
        // not three. A caller pricing a preference is choosing that curve as well as its scale, and
        // reporting a linear price here would misstate what the solver actually traded.
        assert_eq!(r.get("soft_cost").unwrap().as_f64(), Some(9.0), "{r:?}");
    }

    #[test]
    fn a_soft_price_that_is_not_a_number_is_refused_rather_than_ignored() {
        // The failure this guards is silent: an unreadable "soft" that falls back to None leaves a
        // HARD constraint where a preference was asked for, which is the opposite instruction,
        // returned with feasible: true and nothing to suggest anything went wrong. Five bugs of
        // exactly this shape have already been found on this surface.
        for bad in [r#""soft":"5""#, r#""soft":true"#, r#""soft":0"#, r#""soft":-3"#] {
            let e = dispatch(
                "solve",
                &crate::json::parse(&format!(
                    r#"{{"variables":[{{"name":"a","values":2}},{{"name":"b","values":2}}],
                        "constraints":[{{"type":"not_equal","a":"a","b":"b",{bad}}}]}}"#
                ))
                .unwrap(),
            )
            .unwrap_err();
            assert!(e.contains("soft"), "{bad} should be refused by name, got: {e}");
        }
    }

    /// The weighted row this surface could not state at all.
    #[test]
    fn a_weighted_linear_row_crosses_the_wire_and_binds() {
        let r = dispatch(
            "solve",
            &crate::json::parse(
                r#"{"variables":[{"name":"a","values":2},{"name":"b","values":2},
                                 {"name":"c","values":2}],
                    "constraints":[{"type":"linear","rel":"<=","rhs":7,"of":[
                       {"var":"a","value":1,"coeff":3},{"var":"b","value":1,"coeff":4},
                       {"var":"c","value":1,"coeff":5}]}],
                    "objective":{"maximize":true,"terms":[
                       {"weight":1,"of":[{"var":"a","value":1}]},
                       {"weight":1,"of":[{"var":"b","value":1}]},
                       {"weight":1,"of":[{"var":"c","value":1}]}]},
                    "tries":32}"#,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(r.get("feasible").unwrap().as_bool(), Some(true), "{r:?}");
        let vals = r.get("values").unwrap();
        let at = |k: &str| vals.get(k).unwrap().as_f64().unwrap();
        let load = 3.0 * at("a") + 4.0 * at("b") + 5.0 * at("c");
        assert_eq!(load, 7.0, "the row BINDS, and 3 + 4 is the best that fits: {r:?}");

        // A row nothing can satisfy is refused by arithmetic, with the reason.
        let e = dispatch(
            "solve",
            &crate::json::parse(
                r#"{"variables":[{"name":"a","values":2},{"name":"b","values":2}],
                    "constraints":[{"type":"linear","rel":">=","rhs":9,"of":[
                       {"var":"a","value":1,"coeff":3},{"var":"b","value":1,"coeff":4}]}]}"#,
            )
            .unwrap(),
        )
        .unwrap_err();
        assert!(e.contains("no answer"), "must say why: {e}");

        // And a relation nobody defined is named rather than assumed.
        let e = dispatch(
            "solve",
            &crate::json::parse(
                r#"{"variables":[{"name":"a","values":2}],
                    "constraints":[{"type":"linear","rel":"<","rhs":1,
                                    "of":[{"var":"a","value":1,"coeff":1}]}]}"#,
            )
            .unwrap(),
        )
        .unwrap_err();
        assert!(e.contains("rel"), "{e}");
    }

    #[test]
    fn all_different_solves_a_latin_square_row_over_the_wire() {
        let r = dispatch(
            "solve",
            &crate::json::parse(
                r#"{"variables":[{"name":"c0","values":4},{"name":"c1","values":4},
                                 {"name":"c2","values":4},{"name":"c3","values":4}],
                    "constraints":[{"type":"all_different","of":[
                       {"var":"c0"},{"var":"c1"},{"var":"c2"},{"var":"c3"}]}],
                    "tries":60}"#,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(r.get("feasible").unwrap().as_bool(), Some(true), "{r:?}");
        let vals = r.get("values").unwrap();
        let mut got: Vec<f64> = ["c0", "c1", "c2", "c3"]
            .iter()
            .map(|k| vals.get(k).unwrap().as_f64().unwrap())
            .collect();
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(got, vec![0.0, 1.0, 2.0, 3.0], "a permutation, every value once: {r:?}");
    }

    #[test]
    fn an_impossible_all_different_is_refused_by_name_not_annealed() {
        // Five variables over three values. Annealing this returns feasible: false, which reads as
        // "raise the penalty" -- advice that cannot work, because no penalty makes it satisfiable.
        let e = dispatch(
            "solve",
            &crate::json::parse(
                r#"{"variables":[{"name":"x0","values":3},{"name":"x1","values":3},
                                 {"name":"x2","values":3},{"name":"x3","values":3},
                                 {"name":"x4","values":3}],
                    "constraints":[{"type":"all_different","of":[
                       {"var":"x0"},{"var":"x1"},{"var":"x2"},{"var":"x3"},{"var":"x4"}]}]}"#,
            )
            .unwrap(),
        )
        .unwrap_err();
        assert!(e.contains("No assignment can satisfy"), "must say why: {e}");
        assert!(e.contains('5') && e.contains('3'), "and name the counts: {e}");
    }

    #[test]
    fn all_different_needs_variables_and_says_so() {
        for (body, want) in [
            (r#""constraints":[{"type":"all_different","of":[{"var":"a"}]}]"#, "at least two"),
            (r#""constraints":[{"type":"all_different"}]"#, "needs"),
            (r#""constraints":[{"type":"all_different","of":[{"value":1},{"var":"b"}]}]"#, "\"var\""),
        ] {
            let e = dispatch(
                "solve",
                &crate::json::parse(&format!(
                    r#"{{"variables":[{{"name":"a","values":3}},{{"name":"b","values":3}}],{body}}}"#
                ))
                .unwrap(),
            )
            .unwrap_err();
            assert!(e.contains(want), "expected {want:?} in: {e}");
        }
    }

    #[test]
    fn an_inexact_encoding_is_reported_over_the_wire() {
        // A caller who picks a binary encoding for a k that is not a power of two gets an answer
        // that looks fine. The spare codewords decode to nothing and cost exactly what a valid
        // state costs, so the reply has to say so rather than leaving them to find out.
        let r = dispatch(
            "solve",
            &crate::json::parse(
                r#"{"variables":[{"name":"x","values":6,"encoding":"binary"},
                                 {"name":"y","values":8,"encoding":"binary"},
                                 {"name":"z","values":6}],
                    "tries":4}"#,
            )
            .unwrap(),
        )
        .unwrap();
        let cav = r.get("caveats").unwrap().as_arr().unwrap();
        assert_eq!(cav.len(), 1, "only x is inexact: {r:?}");
        let text = cav[0].as_str().unwrap();
        assert!(text.contains("'x'"), "must name it: {text}");

        // And an exact model says nothing, so the field is a signal rather than noise.
        let clean = dispatch(
            "solve",
            &crate::json::parse(r#"{"variables":[{"name":"a","values":5}],"tries":4}"#).unwrap(),
        )
        .unwrap();
        assert!(clean.get("caveats").unwrap().as_arr().unwrap().is_empty(), "{clean:?}");
    }

    #[test]
    fn the_reply_carries_an_ommx_instance_that_decodes() {
        let r = dispatch(
            "solve",
            &crate::json::parse(
                r#"{"variables":[{"name":"a","values":3},{"name":"b","values":3}],
                    "constraints":[{"type":"not_equal","a":"a","b":"b"}],"tries":2}"#,
            )
            .unwrap(),
        )
        .unwrap();
        let b64s = r.get("ommx_b64").unwrap().as_str().unwrap();
        assert!(!b64s.is_empty());
        assert_eq!(b64s.len() % 4, 0, "base64 must be padded to a multiple of four");
        assert!(r.get("ommx_constant").unwrap().as_f64().is_some());

        // Decode and check it is the instance the library would have produced directly, so the
        // wire path cannot diverge from the in-process one.
        let decoded = from_b64(b64s);
        let compiled_again = {
            let mut m = ferrotherm::model::Model::new();
            let a = m.categorical("a", 3);
            let b = m.categorical("b", 3);
            m.not_equal(a, b);
            m.compile().unwrap()
        };
        let direct = ferrotherm::ommx::export(&compiled_again.graph);
        assert_eq!(decoded, direct.bytes, "the HTTP payload must be the library's own bytes");
    }

    /// Decode base64, for the test above only.
    fn from_b64(s: &str) -> Vec<u8> {
        const A: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let idx = |c: u8| A.iter().position(|&x| x == c).unwrap() as u32;
        let raw: Vec<u8> = s.bytes().filter(|&c| c != b'=').collect();
        let mut out = Vec::new();
        for chunk in raw.chunks(4) {
            let mut n = 0u32;
            for (k, &c) in chunk.iter().enumerate() {
                n |= idx(c) << (18 - 6 * k);
            }
            out.push((n >> 16) as u8);
            if chunk.len() > 2 { out.push((n >> 8) as u8); }
            if chunk.len() > 3 { out.push(n as u8); }
        }
        out
    }
}
