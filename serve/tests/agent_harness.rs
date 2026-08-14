//! Can a caller who has never read this code get useful work done?
//!
//! "Agent friendly" is a claim, and these tests are what make it checkable. Each one walks the
//! path a model actually takes: list the tools, read a schema, build a request from the schema
//! alone, call it, and confirm the answer independently. Nothing here reaches for knowledge that
//! is not reachable through the protocol itself.

use ferrotherm_serve::json::{parse, write, Json};
use ferrotherm_serve::{api, mcp};

fn rpc(line: &str) -> Json {
    parse(&mcp::handle(line).expect("a request must get a response")).unwrap()
}

fn call_tool(name: &str, args: &str) -> Result<Json, String> {
    let req = format!(
        r#"{{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{{"name":"{name}","arguments":{args}}}}}"#
    );
    let r = rpc(&req);
    let res = r.get("result").expect("no result");
    let text = res.get("content").unwrap().as_arr().unwrap()[0]
        .get("text")
        .unwrap()
        .as_str()
        .unwrap()
        .to_string();
    if res.get("isError").unwrap().as_bool() == Some(true) {
        Err(text)
    } else {
        Ok(parse(&text).unwrap())
    }
}

#[test]
fn discovery_alone_is_enough_to_make_a_valid_call() {
    // A model sees only tools/list. Every field it needs to construct a request must be described
    // there, including the shape of the graph argument, which is the one non-obvious input.
    let tools = rpc(r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#);
    let list = tools.get("result").unwrap().get("tools").unwrap().as_arr().unwrap();

    let sample = list.iter().find(|t| t.get("name").unwrap().as_str() == Some("ferrotherm_sample"));
    let sample = sample.expect("ferrotherm_sample must be listed");
    let graph_desc = sample
        .get("inputSchema")
        .unwrap()
        .get("properties")
        .unwrap()
        .get("graph")
        .unwrap()
        .get("description")
        .unwrap()
        .as_str()
        .unwrap();

    // The description has to carry a literal example, not just prose about one.
    assert!(graph_desc.contains("builtin"), "graph schema must show the builtin form");
    assert!(graph_desc.contains("couplings"), "graph schema must show the explicit form");
    assert!(graph_desc.contains("-1/+1"), "graph schema must state the state convention");
    assert!(graph_desc.contains("energy is"), "graph schema must state the energy convention");

    // Lift an example straight out of the description and use it verbatim.
    let start = graph_desc.find("{\"builtin\":\"lattice2d\"").expect("a copyable example");
    let end = graph_desc[start..].find('}').unwrap() + start + 1;
    let lifted = &graph_desc[start..end];
    let r = call_tool("ferrotherm_sample", &format!(r#"{{"graph":{lifted},"sweeps":5}}"#))
        .expect("an example lifted from the schema must actually work");
    assert_eq!(r.get("nodes").unwrap().as_f64(), Some(1024.0));
}

#[test]
fn an_agent_can_solve_a_problem_and_check_its_own_answer() {
    // The task: find the ground state of a frustrated 5-cycle. An odd antiferromagnetic ring
    // cannot be two-coloured, so exactly one bond must stay unsatisfied and the optimum is -3.
    let g = r#"{"n":5,"couplings":[[0,1,-1],[1,2,-1],[2,3,-1],[3,4,-1],[4,0,-1]]}"#;
    let solved = call_tool("ferrotherm_anneal", &format!(r#"{{"graph":{g},"beta_max":5.0}}"#)).unwrap();
    let best = solved.get("best_energy").unwrap().as_f64().unwrap();
    assert_eq!(best, -3.0, "frustrated 5-cycle optimum is -3");

    // Now the part that matters: the agent re-scores its own state with a different tool, rather
    // than believing the number the first tool handed back.
    let state = write(solved.get("state").unwrap());
    let scored = call_tool("ferrotherm_energy", &format!(r#"{{"graph":{g},"state":{state}}}"#)).unwrap();
    assert_eq!(
        scored.get("energy").unwrap().as_f64(),
        Some(best),
        "independent scoring must agree with the annealer"
    );
}

#[test]
fn an_agent_can_establish_trust_before_scaling_up() {
    // The documented workflow: verify on a graph small enough to enumerate, then run big. This
    // test is that workflow, and it fails if the advice does not hold.
    let small = call_tool(
        "ferrotherm_verify",
        r#"{"graph":{"builtin":"ring","n":10,"j":1.0},"beta":0.5,"draws":30000,"thin":4}"#,
    )
    .unwrap();
    let tv = small.get("total_variation_distance").unwrap().as_f64().unwrap();
    let floor = small.get("expected_sampling_noise").unwrap().as_f64().unwrap();
    assert!(tv < floor, "TV {tv} must sit under the reported {floor} noise floor");

    // Trust established, scale to a size where enumeration is impossible.
    let big = call_tool(
        "ferrotherm_sample",
        r#"{"graph":{"builtin":"lattice2d","l":300},"beta":0.5,"sweeps":20,"return_state":false}"#,
    )
    .unwrap();
    assert_eq!(big.get("nodes").unwrap().as_f64(), Some(90_000.0));
    assert!(big.get("state").is_none(), "a 90k state should not be dumped unasked");
}

#[test]
fn every_error_message_tells_the_caller_what_to_do_next() {
    // A model recovers from a bad call only if the message names the fix. Each of these is a
    // mistake a first attempt actually makes.
    let cases: Vec<(&str, &str, &str)> = vec![
        ("ferrotherm_sample", r#"{}"#, "graph"),
        ("ferrotherm_sample", r#"{"graph":{"builtin":"grid","l":4}}"#, "lattice2d"),
        ("ferrotherm_sample", r#"{"graph":{"n":3,"couplings":[[0,5,1.0]]}}"#, "out of range"),
        ("ferrotherm_sample", r#"{"graph":{"n":3,"couplings":[[0,1]]}}"#, "3 entries"),
        ("ferrotherm_energy", r#"{"graph":{"builtin":"ring","n":4},"state":[1,-1]}"#, "4 nodes"),
        ("ferrotherm_energy", r#"{"graph":{"builtin":"ring","n":2},"state":[1,0]}"#, "-1 or +1"),
        ("ferrotherm_verify", r#"{"graph":{"builtin":"lattice2d","l":8}}"#, "capped at 20"),
    ];
    for (tool, args, must_say) in cases {
        let e = call_tool(tool, args).expect_err(&format!("{args} should fail"));
        assert!(
            e.contains(must_say),
            "error for {args}\n  said: {e}\n  must mention: {must_say}"
        );
    }
}

#[test]
fn capabilities_describes_the_server_without_recourse_to_source() {
    let c = api::capabilities();
    for key in ["name", "version", "description", "operations", "graph_spec", "limits"] {
        assert!(c.get(key).is_some(), "capabilities is missing {key}");
    }
    // Every advertised operation must be callable by the name given.
    for op in c.get("operations").unwrap().as_arr().unwrap() {
        let name = op.get("name").unwrap().as_str().unwrap();
        let e = api::dispatch(name, &Json::Obj(Vec::new()));
        assert!(
            !matches!(&e, Err(m) if m.starts_with("unknown operation")),
            "{name} is advertised but not dispatchable"
        );
    }
    // The terminology note keeps the naming straight for anyone reading tool output.
    let t = c.get("unit_terminology").unwrap().as_str().unwrap();
    assert!(t.contains("binary stochastic neuron") && t.contains("p-bit"));
}

#[test]
fn the_two_transports_return_the_same_answer() {
    // The reason api:: exists as one module. If HTTP and MCP ever diverge, this catches it.
    use ferrotherm_serve::http::{route, Request};
    let body = r#"{"graph":{"builtin":"ring","n":16},"beta":0.7,"sweeps":40,"seed":11}"#;
    let via_http = parse(
        &route(&Request {
            method: "POST".into(),
            path: "/v1/sample".into(),
            body: body.into(),
        })
        .body,
    )
    .unwrap();
    let via_mcp = call_tool("ferrotherm_sample", body).unwrap();
    for k in ["state", "energy", "magnetization", "nodes", "beta", "seed"] {
        assert_eq!(via_http.get(k), via_mcp.get(k), "{k} differs between transports");
    }
}

/// Everything a solve-shaped task needs, discovered from the protocol and nothing else.
///
/// The harness had no test here at all, which is how a whole family of defects on this tool -- a
/// maximize flag that inverted the objective, a value that silently became zero, an inequality at
/// its boundary that compiled to nothing, a `feasible` that did not check the constraints -- lived
/// behind a green suite. The tool an agent is told to reach for first was the one nothing drove.
#[test]
fn an_agent_can_state_and_solve_a_problem_from_the_schema_alone() {
    // 1. The handshake must point at it. An agent reads this once and plans from it.
    let init = rpc(
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}"#,
    );
    let instructions = init
        .get("result")
        .and_then(|r| r.get("instructions"))
        .and_then(|s| s.as_str())
        .unwrap_or("");
    assert!(
        instructions.contains("ferrotherm_solve"),
        "the handshake has to name the tool it wants used first: {instructions}"
    );

    // 2. Find it in the listing and read its schema.
    let tools = rpc(r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#);
    let list = tools.get("result").unwrap().get("tools").unwrap().as_arr().unwrap();
    let solve = list
        .iter()
        .find(|t| t.get("name").and_then(|n| n.as_str()) == Some("ferrotherm_solve"))
        .expect("ferrotherm_solve must appear in tools/list");
    let props = solve
        .get("inputSchema")
        .and_then(|s| s.get("properties"))
        .and_then(|p| p.as_obj())
        .expect("a schema with properties");
    let named: Vec<&str> = props.iter().map(|(k, _)| k.as_str()).collect();
    for want in ["variables", "constraints", "objective", "tries", "penalty", "schedule"] {
        assert!(named.contains(&want), "the schema must advertise {want:?}: {named:?}");
    }

    // 3. Build a request from the schema's own wording and call it. Graph colouring, because the
    //    answer is checkable without trusting the solver.
    let r = call_tool(
        "ferrotherm_solve",
        r#"{"variables":[{"name":"west","values":3},{"name":"middle","values":3},
                         {"name":"east","values":3}],
            "constraints":[{"type":"not_equal","a":"west","b":"middle"},
                           {"type":"not_equal","a":"middle","b":"east"},
                           {"type":"not_equal","a":"west","b":"east"}],
            "tries":12}"#,
    )
    .unwrap();
    assert_eq!(r.get("feasible").unwrap().as_bool(), Some(true), "{}", write(&r));
    let v = r.get("values").unwrap();
    let got: Vec<f64> = ["west", "middle", "east"]
        .iter()
        .map(|k| v.get(k).unwrap().as_f64().unwrap())
        .collect();
    assert!(
        got[0] != got[1] && got[1] != got[2] && got[0] != got[2],
        "a triangle needs three colours: {got:?}"
    );
    assert!(r.get("violated").unwrap().as_arr().unwrap().is_empty());

    // 4. The schema says "maximize" takes true or 1. Both must mean the same thing, because an
    //    agent writing JSON by hand will produce either.
    let pick = |maximize: &str| -> f64 {
        let r = call_tool(
            "ferrotherm_solve",
            &format!(
                r#"{{"variables":[{{"name":"x","values":5}}],
                     "objective":{{"maximize":{maximize},
                                   "terms":[{{"var":"x","value":4,"weight":5}}]}},"tries":10}}"#
            ),
        )
        .unwrap();
        r.get("values").unwrap().get("x").unwrap().as_f64().unwrap()
    };
    assert_eq!(pick("true"), 4.0, "the reward is at 4");
    assert_eq!(pick("1"), pick("true"), "and the integer form is not the opposite request");

    // 5. An infeasible answer has to be self-explaining: which constraint, and what to do.
    let r = call_tool(
        "ferrotherm_solve",
        r#"{"variables":[{"name":"a","values":3},{"name":"b","values":3}],
            "constraints":[{"type":"not_equal","a":"a","b":"b"}],
            "objective":{"maximize":true,
                         "terms":[{"var":"a","value":1,"weight":40},
                                  {"var":"b","value":1,"weight":40}]},
            "penalty":1,"tries":12}"#,
    )
    .unwrap();
    assert_eq!(r.get("feasible").unwrap().as_bool(), Some(false), "{}", write(&r));
    let broken = r.get("violated").unwrap().as_arr().unwrap();
    assert_eq!(broken.len(), 1, "{}", write(&r));
    // Each violation is an object now: what broke, and by how much. The magnitude is what tells a
    // near miss from a rout, and an agent deciding whether to raise the penalty or restructure the
    // model needs it.
    let first = &broken[0];
    let what = first.get("constraint").and_then(|c| c.as_str()).unwrap_or("");
    assert!(what.contains("must differ"), "it says which: {what}");
    assert!(
        first.get("by").and_then(|b| b.as_f64()).unwrap_or(0.0) > 0.0,
        "and by how much: {}",
        write(first)
    );
    assert!(
        r.get("note").unwrap().as_str().unwrap().contains("penalty"),
        "the note has to name the remedy"
    );

    // 6. Follow that advice from inside the protocol and watch it work.
    let fixed = call_tool(
        "ferrotherm_solve",
        r#"{"variables":[{"name":"a","values":3},{"name":"b","values":3}],
            "constraints":[{"type":"not_equal","a":"a","b":"b"}],
            "objective":{"maximize":true,
                         "terms":[{"var":"a","value":1,"weight":40},
                                  {"var":"b","value":1,"weight":40}]},
            "penalty":500,"tries":12}"#,
    )
    .unwrap();
    assert_eq!(fixed.get("feasible").unwrap().as_bool(), Some(true), "{}", write(&fixed));

    // 7. A bad call must teach rather than merely refuse.
    let e = call_tool(
        "ferrotherm_solve",
        r#"{"variables":[{"name":"t","lo":10,"hi":20}],
            "constraints":[{"type":"fix","var":"t","value":3}]}"#,
    )
    .unwrap_err();
    assert!(e.contains("10..=20"), "an out-of-range value names the range: {e}");
}

/// Every parameter the tools' own output tells you to reach for must be in their schemas.
///
/// `ferrotherm_verify` returns a note saying to raise `thin`, and `thin` was not advertised
/// anywhere. An agent that cannot find the remedy in the schema cannot apply it.
#[test]
fn the_remedy_a_tool_recommends_is_a_parameter_it_advertises() {
    let tools = rpc(r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#);
    let list = tools.get("result").unwrap().get("tools").unwrap().as_arr().unwrap();
    let schema_of = |name: &str| -> Vec<String> {
        list.iter()
            .find(|t| t.get("name").and_then(|n| n.as_str()) == Some(name))
            .and_then(|t| t.get("inputSchema"))
            .and_then(|s| s.get("properties"))
            .and_then(|p| p.as_obj())
            .map(|p| p.iter().map(|(k, _)| k.clone()).collect())
            .unwrap_or_default()
    };

    let r = call_tool(
        "ferrotherm_verify",
        r#"{"graph":{"builtin":"ring","n":8,"j":1.0},"beta":1.6,"draws":20000}"#,
    )
    .unwrap();
    let note = r.get("note").and_then(|n| n.as_str()).unwrap_or("");
    let verify_props = schema_of("ferrotherm_verify");
    for word in ["thin", "draws", "sweeps"] {
        if note.contains(word) {
            assert!(
                verify_props.iter().any(|p| p == word),
                "verify's note recommends {word:?} and its schema does not offer it: {verify_props:?}"
            );
        }
    }

    let r = call_tool(
        "ferrotherm_solve",
        r#"{"variables":[{"name":"a","values":3},{"name":"b","values":3}],
            "constraints":[{"type":"not_equal","a":"a","b":"b"}],"tries":8}"#,
    )
    .unwrap();
    let note = r.get("note").and_then(|n| n.as_str()).unwrap_or("");
    let solve_props = schema_of("ferrotherm_solve");
    for word in ["penalty", "schedule"] {
        if note.contains(word) {
            assert!(
                solve_props.iter().any(|p| p == word),
                "solve's note recommends {word:?} and its schema does not offer it: {solve_props:?}"
            );
        }
    }
}
