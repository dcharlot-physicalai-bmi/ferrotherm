"""Solve the agreed model through the MCP server and print the answer.

Its own file rather than a heredoc inside `check-answers.sh`: the script already nests one, and a
second whose terminator is also `PY` silently closes the outer one. Separate files also mean the
driver can be run by hand when a surface disagrees, which is when you most want to.

Prints nothing on failure, so the gate reports PRODUCED NOTHING and shows stderr.
"""

import json
import os
import subprocess
import sys

ARGS = {
    "variables": [
        {"name": "a", "values": 3},
        {"name": "b", "values": 3},
        {"name": "t", "lo": 10, "hi": 13},
    ],
    "constraints": [
        {"type": "not_equal", "a": "a", "b": "b"},
        {"type": "at_most", "k": 1, "of": [{"var": "a", "value": 0}, {"var": "b", "value": 0}]},
        {"type": "fix", "var": "t", "value": 12},
    ],
    "objective": {
        "maximize": True,
        "terms": [
            {"var": "a", "value": 1, "weight": 3},
            {"var": "b", "value": 2, "weight": 4},
        ],
    },
    "tries": 64,
}

REQS = [
    {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "check-answers", "version": "1"},
        },
    },
    {
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": "ferrotherm_solve", "arguments": ARGS},
    },
]


def main() -> int:
    out_dir = sys.argv[1]
    proc = subprocess.run(
        ["./target/release/ferrotherm-mcp"],
        input="".join(json.dumps(r) + "\n" for r in REQS),
        capture_output=True,
        text=True,
        timeout=180,
    )
    for line in proc.stdout.splitlines():
        try:
            msg = json.loads(line)
        except Exception:
            continue
        if msg.get("id") != 2:
            continue
        payload = json.loads(msg["result"]["content"][0]["text"])
        v = payload["values"]
        answer = (
            f"a={v['a']} b={v['b']} t={v['t']} "
            f"feasible={str(payload['feasible']).lower()}"
        )
        with open(os.path.join(out_dir, "mcp.txt"), "w") as fh:
            fh.write(answer)
        return 0
    print("no tools/call result in the MCP response", file=sys.stderr)
    print(proc.stderr[:400], file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
