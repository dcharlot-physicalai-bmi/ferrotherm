//! The only check that survives me having misread the schema: have the REFERENCE implementation
//! read what this crate writes, and score it.
//!
//! The unit tests in `lib.rs` verify my own arithmetic against ferrotherm's energy. They would pass
//! just as happily if every protobuf field number were wrong, because they never leave this process.
//! This test shells out to Python's `ommx` -- a different language, a different codebase, the format's
//! own maintainers -- and asks it to evaluate the instance. It skips when that is not installed,
//! because a missing optional interpreter is not a defect in this crate.
//!
//!     python3 -m venv /tmp/ommxref && /tmp/ommxref/bin/pip install ommx
//!     OMMX_PYTHON=/tmp/ommxref/bin/python cargo test -p ferrotherm-ommx

use std::io::Write;
use std::process::{Command, Stdio};

#[test]
fn the_reference_implementation_scores_every_state_as_we_do() {
    let py = std::env::var("OMMX_PYTHON").unwrap_or_else(|_| "python3".into());
    let has = Command::new(&py)
        .args(["-c", "import ommx.v1"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !has {
        eprintln!("no `ommx` for {py}; skipping the reference check");
        return;
    }

    let g = ferrotherm::ising::lattice2d(3, 1.0);
    let export = ferrotherm_ommx::export(&g);
    let dir = std::env::temp_dir().join("ferrotherm-ommx-ref");
    std::fs::create_dir_all(&dir).unwrap();
    let inst = dir.join("instance.ommx");
    std::fs::write(&inst, &export.bytes).unwrap();

    // Every state, with the energy ferrotherm assigns it.
    let mut want = String::new();
    for mask in 0..(1u32 << g.n) {
        let s: Vec<i8> = (0..g.n).map(|i| if (mask >> i) & 1 == 1 { 1 } else { -1 }).collect();
        want.push_str(&format!("{mask} {}\n", g.energy(&s)));
    }
    let table = dir.join("energies.txt");
    std::fs::write(&table, &want).unwrap();

    let script = r#"
import sys
from ommx.v1 import Instance, State
inst = Instance.from_bytes(open(sys.argv[1], "rb").read())
n = len(inst.decision_variables)
bad = 0
for line in open(sys.argv[2]):
    mask, e = line.split(); mask, e = int(mask), float(e)
    x = {i: float((mask >> i) & 1) for i in range(n)}
    if abs(inst.evaluate(State(entries=x)).objective - e) > 1e-9:
        bad += 1
print(f"{n} {bad}")
"#;
    let mut child = Command::new(&py)
        .args(["-c", script, inst.to_str().unwrap(), table.to_str().unwrap()])
        .stdout(Stdio::piped())
        .stdin(Stdio::piped())
        .spawn()
        .expect("spawn python");
    child.stdin.take().map(|mut s| s.write_all(b"").ok());
    let out = child.wait_with_output().expect("run python");
    let text = String::from_utf8_lossy(&out.stdout);
    let mut it = text.split_whitespace();
    let vars: usize = it.next().and_then(|v| v.parse().ok()).unwrap_or(0);
    let bad: usize = it.next().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);

    assert_eq!(vars, g.n, "the reference must see one decision variable per spin");
    assert_eq!(
        bad, 0,
        "the reference implementation scored {bad} of {} states differently from ferrotherm",
        1u32 << g.n
    );
}
