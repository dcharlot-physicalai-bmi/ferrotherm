use ferrotherm_serve::{api, json};
use std::time::Instant;

fn main() {
    // The ceiling check is `nodes * sweeps`. The draw phase (`draws * thin` sweeps) is not counted.
    let body = r#"{"graph":{"builtin":"ring","n":1000},"sweeps":1,"draws":100,"thin":1000,
                   "return_state":false}"#;
    let t0 = Instant::now();
    let r = api::dispatch("sample", &json::parse(body).unwrap()).unwrap();
    let led = r.get("ledger").unwrap();
    println!("budget the ceiling checked : {}", 1000u64 * 1);
    println!("node updates actually done : {:?}", led.get("node_updates").unwrap().as_f64());
    println!("wall {:.2}s   MAX_NODE_UPDATES = {}", t0.elapsed().as_secs_f64(), api::MAX_NODE_UPDATES);

    // the same total work stated as `sweeps` IS refused
    let refused = r#"{"graph":{"builtin":"lattice2d","l":2000},"sweeps":6000}"#;
    println!("as sweeps  -> {:?}", api::dispatch("sample", &json::parse(refused).unwrap()).err());
    // stated as draws x thin the identical work passes the check (not run here: it would take hours)
}
