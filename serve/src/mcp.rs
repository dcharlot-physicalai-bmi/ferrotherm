//! Model Context Protocol server over stdio: JSON-RPC 2.0, one message per line.
//!
//! The tools are the same operations the HTTP API exposes, described with JSON Schema so a caller
//! can use them without being told how.

use crate::api;
use crate::json::{parse, write, Json};
use std::io::{BufRead, Write as _};

/// The protocol revision this server implements. A client asking for a different one is answered
/// with theirs when we can honour it, which is what the specification asks for.
pub const PROTOCOL: &str = "2025-06-18";
const SUPPORTED: [&str; 3] = ["2025-06-18", "2025-03-26", "2024-11-05"];

fn schema_graph() -> Json {
    Json::obj(vec![
        ("type", Json::s("object")),
        (
            "description",
            Json::s(
                "Either a named family or an explicit edge list. Named: \
                 {\"builtin\":\"lattice2d\",\"l\":32,\"j\":1.0} or \
                 {\"builtin\":\"ring\",\"n\":16,\"j\":1.0,\"h\":0.0}. Explicit: \
                 {\"n\":4,\"couplings\":[[0,1,1.0]],\"biases\":[[0,0.5]]}. States are -1/+1 and \
                 energy is -sum_ij J_ij s_i s_j - sum_i h_i s_i.",
            ),
        ),
    ])
}

fn prop(t: &str, desc: &str) -> Json {
    Json::obj(vec![("type", Json::s(t)), ("description", Json::s(desc))])
}

fn tool(name: &str, desc: &str, props: Vec<(&str, Json)>, required: Vec<&str>) -> Json {
    Json::obj(vec![
        ("name", Json::s(name)),
        ("description", Json::s(desc)),
        (
            "inputSchema",
            Json::obj(vec![
                ("type", Json::s("object")),
                ("properties", Json::obj(props)),
                ("required", Json::Arr(required.into_iter().map(Json::s).collect())),
            ]),
        ),
    ])
}

pub fn tools() -> Json {
    Json::Arr(vec![
        tool(
            "ferrotherm_sample",
            "Draw a state from the Boltzmann distribution of an Ising graph by chromatic \
             block-Gibbs sampling. Returns the state, its energy, magnetization, an energy ledger \
             priced at Z1-class device figures, and a CERTIFICATE computed from the samples \
             themselves: the temperature actually sampled at with a confidence interval, the \
             autocorrelation time and effective sample size, and where the model is small enough, \
             the distance from the exact Boltzmann distribution beside the sampling-noise floor. \
             An empty findings list is the only thing that means the run is sound. Deterministic \
             given seed and thread count.",
            vec![
                ("graph", schema_graph()),
                ("beta", prop("number", "Inverse temperature, 1/T. Higher is colder and more ordered. Default 1.0.")),
                ("sweeps", prop("integer", "Burn-in sweeps before recording. Default 100.")),
                ("draws", prop("integer", "Samples recorded after burn-in, used to certify the run. Default 128.")),
                ("thin", prop("integer", "Sweeps between recorded draws. Raise this if the certificate reports correlated draws. Default 1.")),
                ("seed", prop("integer", "Random seed. Default 0.")),
                ("threads", prop("integer", "Parallel threads. Results stay reproducible per thread count. Default 1.")),
                ("clamp", prop("array", "Nodes held fixed, as [[index, -1 or +1], ...].")),
                ("return_state", prop("boolean", "Include the full state array. Defaults to true for graphs of 4096 nodes or fewer.")),
            ],
            vec!["graph"],
        ),
        tool(
            "ferrotherm_anneal",
            "Minimise the energy of an Ising graph by simulated annealing down a geometric beta \
             ladder, returning the lowest-energy state found. Use this for optimisation problems \
             encoded as couplings.",
            vec![
                ("graph", schema_graph()),
                ("beta_min", prop("number", "Starting (hot) inverse temperature. Default 0.1.")),
                ("beta_max", prop("number", "Final (cold) inverse temperature. Default 3.0.")),
                ("stages", prop("integer", "Rungs on the ladder. Default 24.")),
                ("sweeps_per_stage", prop("integer", "Sweeps at each rung. Default 20.")),
                ("seed", prop("integer", "Random seed. Default 0.")),
                ("return_state", prop("boolean", "Include the full state array.")),
            ],
            vec!["graph"],
        ),
        tool(
            "ferrotherm_energy",
            "Compute the energy and magnetization of a specific state under a specific graph. Use \
             it to score a candidate solution.",
            vec![
                ("graph", schema_graph()),
                ("state", prop("array", "One entry per node, each -1 or +1.")),
            ],
            vec!["graph", "state"],
        ),
        tool(
            "ferrotherm_verify",
            "Check the sampler against ground truth: enumerate the exact Boltzmann distribution \
             and report the total variation distance from the sampler's empirical one. Capped at \
             20 nodes because enumeration is 2^n. Compare the result against the reported sampling \
             noise floor rather than against zero.",
            vec![
                ("graph", schema_graph()),
                ("beta", prop("number", "Inverse temperature. Default 1.0.")),
                ("draws", prop("integer", "Samples to histogram. Default 20000.")),
                ("sweeps", prop("integer", "Burn-in sweeps before sampling. Default 200.")),
                ("thin", prop("integer", "Sweeps between recorded draws. Default 1. Raise this when the reported TV sits above the noise floor: consecutive sweeps are correlated, and thinning is what buys independent draws.")),
                ("seed", prop("integer", "Random seed. Default 0.")),
            ],
            vec!["graph"],
        ),
        tool(
            "ferrotherm_solve",
            "State a constraint or optimisation problem in its own vocabulary and get NAMED VALUES \
             back. Prefer this over ferrotherm_anneal: that one wants a graph of spins and returns \
             an array of +/-1, which means computing spin indices yourself to say something as \
             simple as \"these two must differ\". Declare variables with domains, state \
             constraints, optionally give an objective, and read the answer by name. Check \
             `feasible` before trusting the values. False means the answer breaks something you \
             asked for, and the reply says which: a variable in \"did_not_decode\", or a \
             HARD constraint in \"violated\" that the objective outbid. A penalty makes a \
             constraint expensive, not impossible -- so raise \"penalty\" and try again, and only \
             conclude the problem has no answer if it stays infeasible at a large one. Entries with \
             \"hard\": false are preferences you priced with \"soft\" and do NOT make the answer \
             infeasible; \"soft_cost\" totals what those trades cost. The reply also carries \"caveats\": things the compiler KNOWS are wrong with the model and cannot fix, today meaning an encoding no penalty can make exact (a binary encoding of k values spells 2^ceil(log2 k) codewords, and when k is not a power of two the spare ones decode to nothing while costing exactly what a valid state costs). An empty list is the normal case; a non-empty one means a value that reads back fine may have come from a codeword the penalty never excluded, so report it rather than trusting the number.",
            vec![
                ("variables", prop("array", "Each is {\"name\": \"x\", \"values\": 3} for a categorical, or {\"name\": \"t\", \"lo\": 0, \"hi\": 9} for an integer. Everywhere a \"value\" appears below it is the variable's OWN value, never a slot index: for an integer over 10..20, write 13 to mean 13. A value outside the range is an error naming the range, not a silent reinterpretation. Either kind may carry \"encoding\": how it is STORED -- \"one-hot\" (default, exact, k spins), \"domain-wall\" (exact, k-1 spins, cheaper for large k), or \"binary\" (ceil(log2 k) spins, and NOT exact unless k is a power of two: the spare codewords decode to nothing and cost exactly what a valid state costs, which is reported in \"caveats\"). An unknown name is refused rather than defaulted.")),
                ("constraints", prop("array", "Each is {\"type\": \"not_equal\"|\"equal\", \"a\": name, \"b\": name}, {\"type\": \"fix\", \"var\": name, \"value\": v}, or {\"type\": \"cardinality\"|\"at_most\"|\"at_least\", \"k\": n, \"of\": [{\"var\": name, \"value\": v}, ...]}, or {\"type\": \"exactly_one\"|\"at_most_one\", \"of\": [...]}, or {\"type\": \"all_different\", \"of\": [{\"var\": name}, ...]} which makes every named VARIABLE take a different value -- the workhorse of assignment, scheduling, colouring and puzzles. all_different lowers per shared value rather than per pair, so it needs no slack, costs nothing where two domains do not overlap, and its violation names which value collided and who took it; asking it of more variables than the values they share is refused when the model compiles, because that has no answer at any penalty. Cardinality is exactly k; at_most and at_least are inequalities and cost extra spins, because an inequality needs a slack variable to become an equality the sampler can square. exactly_one and at_most_one lower pairwise instead and need no slack, so prefer them over k=1. \"of\" takes any number of entries, and each may name a different variable AND a different value. ANY constraint may carry \"soft\": w, which makes it a PREFERENCE priced at w rather than a rule: the solver may trade it away, the answer stays feasible, and the reply reports it under \"violated\" with \"hard\": false plus a \"cost\". Use soft for what you would LIKE, hard for what an answer must satisfy to be an answer at all. The price is w x amount SQUARED, so missing by two costs four times missing by one, and unlike \"penalty\" it is absolute rather than scaled against the objective -- that is the point, since a preference is meant to be traded against it.")),
                ("objective", prop("object", "{\"maximize\": true, \"terms\": [{\"var\": name, \"value\": v, \"weight\": w}]}. A term with \"and_var\"/\"and_value\" rewards two variables taking values together, and a term with \"of\": [{\"var\": name, \"value\": v}, ...] rewards ANY number of them holding together. Three or more is a higher-order term: it is lowered with an extra spin per substituted pair, reported as \"ancillas\" in the reply, and its guarantee is about finding optima rather than about sampling. \"maximize\" takes true/false or 1/0; anything else is refused rather than read as false.")),
                ("tries", prop("integer", "Independent annealing restarts; the best feasible answer wins. Default 16.")),
                ("penalty", prop("number", "Constraint strength. Omit it: by default it scales to twice the largest objective weight, because a constraint that ties with an objective gets traded away. Raise it when \"violated\" comes back non-empty.")),
                ("schedule", prop("object", "The annealing ladder: {\"beta_hot\": 0.05, \"beta_cold\": 8.0, \"stages\": 120, \"sweeps\": 40}, which are the defaults. Lengthen it -- more stages, more sweeps -- when a model stays infeasible at a large penalty: that is a model not being annealed long enough, and no penalty fixes it. Cooling runs hot to cold, so beta_cold must exceed beta_hot.")),
            ],
            vec!["variables"],
        ),
        tool(
            "ferrotherm_capabilities",
            "Describe this server: operations, graph specification, limits, and conventions.",
            vec![],
            vec![],
        ),
    ])
}

fn ok(id: Json, result: Json) -> Json {
    Json::obj(vec![("jsonrpc", Json::s("2.0")), ("id", id), ("result", result)])
}

fn rpc_err(id: Json, code: i32, msg: &str) -> Json {
    Json::obj(vec![
        ("jsonrpc", Json::s("2.0")),
        ("id", id),
        ("error", Json::obj(vec![("code", Json::n(code as f64)), ("message", Json::s(msg))])),
    ])
}

/// A tool result carries its payload as text content; an error sets `isError` rather than failing
/// the call, so the model sees what went wrong and can correct the arguments itself.
fn tool_result(text: String, is_error: bool) -> Json {
    Json::obj(vec![
        (
            "content",
            Json::Arr(vec![Json::obj(vec![("type", Json::s("text")), ("text", Json::Str(text))])]),
        ),
        ("isError", Json::Bool(is_error)),
    ])
}

/// Handle one JSON-RPC message. Returns None for notifications, which take no reply.
pub fn handle(line: &str) -> Option<String> {
    let msg = match parse(line) {
        Ok(m) => m,
        Err(e) => return Some(write(&rpc_err(Json::Null, -32700, &format!("parse error: {e}")))),
    };
    let id = msg.get("id").cloned().unwrap_or(Json::Null);
    let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");
    let params = msg.get("params").cloned().unwrap_or(Json::Obj(Vec::new()));

    // Notifications carry no id and must not be answered.
    msg.get("id")?;

    let resp = match method {
        "initialize" => {
            let want = params.get("protocolVersion").and_then(|v| v.as_str()).unwrap_or(PROTOCOL);
            let use_ver = if SUPPORTED.contains(&want) { want } else { PROTOCOL };
            ok(
                id,
                Json::obj(vec![
                    ("protocolVersion", Json::s(use_ver)),
                    ("capabilities", Json::obj(vec![("tools", Json::obj(vec![]))])),
                    (
                        "serverInfo",
                        Json::obj(vec![
                            ("name", Json::s("ferrotherm")),
                            ("version", Json::s(env!("CARGO_PKG_VERSION"))),
                        ]),
                    ),
                    (
                        "instructions",
                        Json::s(
                            "Thermodynamic sampling over Ising graphs, and a modelling layer above \
                             it. Reach for ferrotherm_solve first: state a problem as variables, \
                             constraints and an objective, and read the answer back in your own \
                             names. The others work in spins -- ferrotherm_anneal to optimise a \
                             graph you have already encoded, ferrotherm_sample to draw from a \
                             distribution, ferrotherm_verify to check the sampler against exact \
                             enumeration on small graphs before trusting it on large ones. \
                             Whatever you call, check the result's own verdict field before \
                             trusting its numbers: `feasible` on a solve, `findings` on a verify.",
                        ),
                    ),
                ]),
            )
        }
        "ping" => ok(id, Json::obj(vec![])),
        "tools/list" => ok(id, Json::obj(vec![("tools", tools())])),
        "tools/call" => {
            let name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
            let args = params.get("arguments").cloned().unwrap_or(Json::Obj(Vec::new()));
            let op = name.strip_prefix("ferrotherm_").unwrap_or(name);
            match api::dispatch(op, &args) {
                Ok(r) => ok(id, tool_result(write(&r), false)),
                Err(e) => ok(id, tool_result(e, true)),
            }
        }
        "" => rpc_err(id, -32600, "invalid request: no method"),
        other => rpc_err(id, -32601, &format!("method not found: {other}")),
    };
    Some(write(&resp))
}

/// Read newline-delimited JSON-RPC from stdin until it closes.
pub fn serve() -> std::io::Result<()> {
    let stdin = std::io::stdin();
    let mut out = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if let Some(resp) = handle(&line) {
            writeln!(out, "{resp}")?;
            out.flush()?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(line: &str) -> Json {
        parse(&handle(line).expect("expected a response")).unwrap()
    }

    #[test]
    fn initialize_reports_tools() {
        let r = call(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}"#);
        let res = r.get("result").unwrap();
        assert_eq!(res.get("protocolVersion").unwrap().as_str(), Some("2025-06-18"));
        assert!(res.get("capabilities").unwrap().get("tools").is_some());
    }

    #[test]
    fn honours_an_older_client_protocol() {
        let r = call(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05"}}"#);
        assert_eq!(
            r.get("result").unwrap().get("protocolVersion").unwrap().as_str(),
            Some("2024-11-05")
        );
    }

    #[test]
    fn unknown_protocol_falls_back_to_ours() {
        let r = call(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"1999-01-01"}}"#);
        assert_eq!(r.get("result").unwrap().get("protocolVersion").unwrap().as_str(), Some(PROTOCOL));
    }

    #[test]
    fn every_tool_has_a_usable_schema() {
        let r = call(r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#);
        let ts = r.get("result").unwrap().get("tools").unwrap().as_arr().unwrap();
        assert_eq!(ts.len(), 6);
        for t in ts {
            let name = t.get("name").unwrap().as_str().unwrap();
            assert!(name.starts_with("ferrotherm_"), "{name}");
            assert!(t.get("description").unwrap().as_str().unwrap().len() > 40, "{name} description too thin");
            let s = t.get("inputSchema").unwrap();
            assert_eq!(s.get("type").unwrap().as_str(), Some("object"));
            assert!(s.get("properties").is_some(), "{name} has no properties");
            assert!(s.get("required").is_some(), "{name} has no required list");
        }
    }

    #[test]
    fn tools_call_runs_the_sampler() {
        let r = call(
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"ferrotherm_sample",
                "arguments":{"graph":{"builtin":"ring","n":8},"sweeps":20}}}"#,
        );
        let res = r.get("result").unwrap();
        assert_eq!(res.get("isError").unwrap().as_bool(), Some(false));
        let text = res.get("content").unwrap().as_arr().unwrap()[0].get("text").unwrap().as_str().unwrap();
        assert_eq!(parse(text).unwrap().get("nodes").unwrap().as_f64(), Some(8.0));
    }

    #[test]
    fn a_bad_call_is_a_correctable_tool_error_not_a_dead_request() {
        let r = call(
            r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"ferrotherm_sample",
                "arguments":{"graph":{"n":4,"couplings":[[0,99,1.0]]}}}}"#,
        );
        // the model must still get a result it can read and retry from
        let res = r.get("result").unwrap();
        assert_eq!(res.get("isError").unwrap().as_bool(), Some(true));
        assert!(r.get("error").is_none(), "should not be a protocol error");
        let text = res.get("content").unwrap().as_arr().unwrap()[0].get("text").unwrap().as_str().unwrap();
        assert!(text.contains("out of range"), "{text}");
    }

    #[test]
    fn notifications_get_no_reply() {
        assert!(handle(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#).is_none());
    }

    #[test]
    fn malformed_input_is_a_parse_error_not_a_crash() {
        let r = call("{ this is not json");
        assert_eq!(r.get("error").unwrap().get("code").unwrap().as_f64(), Some(-32700.0));
    }

    #[test]
    fn unknown_method_is_32601() {
        let r = call(r#"{"jsonrpc":"2.0","id":9,"method":"resources/list"}"#);
        assert_eq!(r.get("error").unwrap().get("code").unwrap().as_f64(), Some(-32601.0));
    }

    #[test]
    fn tool_names_match_the_dispatch_table() {
        // a tool advertised but not routable is the failure this catches
        let ts = tools();
        for t in ts.as_arr().unwrap() {
            let name = t.get("name").unwrap().as_str().unwrap();
            let op = name.strip_prefix("ferrotherm_").unwrap();
            let e = api::dispatch(op, &Json::Obj(Vec::new()));
            assert!(
                !matches!(&e, Err(m) if m.starts_with("unknown operation")),
                "{name} is advertised but not dispatchable"
            );
        }
    }
}
