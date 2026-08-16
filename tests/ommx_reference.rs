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
//!     OMMX_PYTHON=/tmp/ommxref/bin/python cargo test --test ommx_reference

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
    let export = ferrotherm::ommx::export(&g);
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

#[test]
fn we_can_read_an_instance_the_reference_built() {
    // The direction that makes this a BRIDGE rather than an exporter: an instance the reference
    // stack produced end to end, which this crate never wrote a byte of.
    //
    // This is what caught the proto3 trap. A field at its default value is not serialised, so
    // `Term { id: 0, coefficient: c }` writes the coefficient alone -- and this crate's reader used
    // a sentinel for "no id seen", turning variable 0 into "not declared". Its own encoder writes
    // id 0 explicitly, so reader and writer agreed with each other and were both wrong about the
    // format. Only a file from somebody else's encoder could show that.
    let py = std::env::var("OMMX_PYTHON").unwrap_or_else(|_| "python3".into());
    let has = Command::new(&py)
        .args(["-c", "import ommx.v1"])
        .stdout(Stdio::null()).stderr(Stdio::null()).status()
        .map(|s| s.success()).unwrap_or(false);
    if !has {
        eprintln!("no `ommx` for {py}; skipping");
        return;
    }

    let dir = std::env::temp_dir().join("ferrotherm-ommx-theirs");
    std::fs::create_dir_all(&dir).unwrap();
    let inst = dir.join("theirs.ommx");
    let table = dir.join("theirs.txt");

    // Deliberately includes a term on variable 0, a negative coefficient, and a constant -- the
    // three things whose omission or sign a hand-rolled reader is most likely to get wrong.
    let build = format!(
        r#"
from ommx.v1 import Instance, DecisionVariable, State
x = [DecisionVariable.binary(i, name=f"q{{i}}") for i in range(4)]
inst = Instance.from_components(decision_variables=x,
    objective=2.0*x[0]*x[1] - 3.0*x[1]*x[2] + 1.5*x[2]*x[3] + 0.5*x[0] - 2.0*x[3] + 7.0,
    constraints=[], sense=Instance.MINIMIZE)
open(r"{}", "wb").write(inst.to_bytes())
rows = []
for m in range(16):
    xs = {{i: float((m >> i) & 1) for i in range(4)}}
    rows.append(f"{{m}} {{inst.evaluate(State(entries=xs)).objective}}")
open(r"{}", "w").write("\n".join(rows))
"#,
        inst.display(), table.display()
    );
    let ok = Command::new(&py).args(["-c", &build]).status().map(|s| s.success()).unwrap_or(false);
    assert!(ok, "the reference failed to build its own instance");

    let (g, constant) = ferrotherm::ommx::import(&std::fs::read(&inst).unwrap())
        .expect("must read what the reference writes");
    assert_eq!(g.n, 4);

    let want = std::fs::read_to_string(&table).unwrap();
    let mut checked = 0;
    for line in want.lines().filter(|l| !l.trim().is_empty()) {
        let mut it = line.split_whitespace();
        let mask: u32 = it.next().unwrap().parse().unwrap();
        let theirs: f64 = it.next().unwrap().parse().unwrap();
        let s: Vec<i8> = (0..g.n).map(|i| if (mask >> i) & 1 == 1 { 1 } else { -1 }).collect();
        let ours = g.energy(&s) + constant;
        assert!(
            (ours - theirs).abs() < 1e-9,
            "state {mask:04b}: the reference says {theirs}, we say {ours}"
        );
        checked += 1;
    }
    assert_eq!(checked, 16, "every state has to be compared, not some of them");
}
