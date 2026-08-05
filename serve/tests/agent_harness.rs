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
