// One model, two notations: does the picture answer what the API answers?
//
// The node editor and the HTTP/MCP API are two ways of saying the same model, and `fromModel`
// claims to carry one into the other without loss. That claim is worth exactly as much as a check
// on it -- a bridge that quietly drops a k, a soft price or an objective term still draws, still
// runs, and answers a different question. So the same JSON goes through both doors and the answers
// are compared.
//
// The comparison is on the COMPILED SIZE and the feasibility, not on the values: both surfaces
// anneal, and two runs of a stochastic sampler on the same model are entitled to differ. Spins and
// ancillas are not entitled to differ -- they are what the model compiled to, and if the editor
// lost a constraint on the way in, this is where it shows.
//
//   node scripts/editor-vs-api.mjs [--probe]
//
// Expects a ferrotherm-serve listening on FT_SERVE (default 127.0.0.1:8479).
import { createRequire } from "node:module";
const REPO = new URL("../", import.meta.url).pathname;
const require = createRequire(import.meta.url);
const PW_PATHS = [REPO, `${REPO}web-tests/`];
function loadPlaywright() {
  for (const base of PW_PATHS) {
    try { return require(require.resolve("playwright", { paths: [base] })); }
    catch (e) { if (e.code !== "MODULE_NOT_FOUND") throw e; }
  }
  throw new Error(`playwright resolves from none of: ${PW_PATHS.map(p => `${p}node_modules`).join(", ")}`);
}
if (process.argv.includes("--probe")) {
  try { loadPlaywright(); process.exit(0); } catch (e) { console.error(e.message); process.exit(1); }
}
const { chromium } = loadPlaywright();
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join } from "node:path";

const API = process.env.FT_SERVE || "127.0.0.1:8479";

// Every constraint kind and both objective shapes, so a surface that lost one has nowhere to hide.
const MODELS = [
  { what: "colouring, stated as all_different",
    m: { variables: [{ name: "west", values: 3 }, { name: "middle", values: 3 }, { name: "east", values: 3 }],
         constraints: [{ type: "all_different", of: [{ var: "west" }, { var: "middle" }, { var: "east" }] }] } },
  { what: "a counting ceiling over nine",
    m: { variables: Array.from({ length: 9 }, (_, i) => ({ name: "s" + i, values: 2 })),
         constraints: [{ type: "at_most", k: 2, of: Array.from({ length: 9 }, (_, i) => ({ var: "s" + i, value: 1 })) }],
         objective: { maximize: true, terms: Array.from({ length: 9 }, (_, i) => ({ var: "s" + i, value: 1, weight: 9 - i })) } } },
  { what: "exactly-one and at-most-one beside each other",
    m: { variables: [{ name: "a", values: 2 }, { name: "b", values: 2 }, { name: "c", values: 2 }, { name: "d", values: 2 }],
         constraints: [{ type: "exactly_one", of: [{ var: "a", value: 1 }, { var: "b", value: 1 }] },
                       { type: "at_most_one", of: [{ var: "c", value: 1 }, { var: "d", value: 1 }] }] } },
  { what: "a priced preference the objective buys out",
    m: { variables: [{ name: "a", values: 2 }, { name: "b", values: 2 }],
         constraints: [{ type: "not_equal", a: "a", b: "b", soft: 1 }],
         objective: { maximize: true, terms: [{ var: "a", value: 1, weight: 20 }, { var: "b", value: 1, weight: 20 }] } } },
  { what: "an integer pinned, and a pair rewarded together",
    m: { variables: [{ name: "t", lo: 10, hi: 14 }, { name: "x", values: 3 }, { name: "y", values: 3 }],
         constraints: [{ type: "fix", var: "t", value: 12 }, { type: "equal", a: "x", b: "y" }],
         objective: { maximize: true, terms: [{ var: "x", value: 2, and_var: "y", and_value: 2, weight: 4 }] } } },
  { what: "an exact count and a floor",
    m: { variables: Array.from({ length: 5 }, (_, i) => ({ name: "v" + i, values: 2 })),
         constraints: [{ type: "cardinality", k: 2, of: [0, 1, 2].map(i => ({ var: "v" + i, value: 1 })) },
                       { type: "at_least", k: 1, of: [3, 4].map(i => ({ var: "v" + i, value: 1 })) }] } },
];

// --selftest serves a DAMAGED editor -- one whose fromModel drops the k off a counting constraint
// -- and expects this gate to fail on it. Six models that have only ever agreed is the same
// evidence as a gate that cannot disagree, and the difference matters: this one compares numbers
// parsed out of a report by regex, and a regex that stops matching reports -1 on both sides and
// calls it agreement.
const SELFTEST = process.argv.includes("--selftest");
const DAMAGE = [/if \(c\.k !== undefined && "k" in n\.vals\) n\.vals\.k = c\.k \| 0;/,
                'if (false) n.vals.k = c.k | 0;'];

const files = createServer(async (req, res) => {
  const p = join(REPO, "docs", (req.url === "/" ? "/graph.html" : req.url).split("?")[0]);
  try {
    let b = await readFile(p);
    if (SELFTEST && p.endsWith("graph.html")) {
      const s = b.toString();
      if (!DAMAGE[0].test(s)) {
        console.error("selftest cannot damage the editor: fromModel no longer carries k that way");
        process.exit(1);
      }
      b = Buffer.from(s.replace(DAMAGE[0], DAMAGE[1]));
    }
    res.writeHead(200, { "content-type": { ".html": "text/html", ".wasm": "application/wasm" }[extname(p)] ?? "application/octet-stream" });
    res.end(b);
  } catch { res.writeHead(404).end("nope"); }
});
await new Promise(r => files.listen(0, r));
const base = `http://localhost:${files.address().port}`;

const browser = await chromium.launch();
const page = await browser.newPage();
const errs = [];
page.on("pageerror", e => errs.push("pageerror: " + e.message));
await page.goto(base + "/graph.html");
await page.evaluate(() => window.ferrotherm.ready);

let fail = 0;
for (const { what, m } of MODELS) {
  let api;
  try {
    const r = await fetch(`http://${API}/v1/solve`, { method: "POST", body: JSON.stringify(m) });
    api = await r.json();
  } catch (e) {
    console.error(`  no server at ${API}: ${e.message}`);
    process.exit(2);
  }
  if (api.error) { console.log(`  FAIL ${what}: the API refused it -- ${api.error}`); fail++; continue; }

  const ed = await page.evaluate(m => {
    const F = window.ferrotherm;
    try {
      F.fromModel(m);
      const txt = F.run();
      return { spins: +(txt.match(/(\d+) spins/)?.[1] ?? -1),
               ancillas: +(txt.match(/ancillas\s+(\d+)/)?.[1] ?? 0),
               feasible: /feasible  yes/.test(txt),
               constraints: +(txt.match(/(\d+) constraints/)?.[1] ?? -1),
               terms: +(txt.match(/(\d+) objective terms/)?.[1] ?? 0),
               txt };
    } catch (e) { return { err: e.message }; }
  }, m);

  if (ed.err) { console.log(`  FAIL ${what}: the editor refused it -- ${ed.err}`); fail++; continue; }

  const same = ed.spins === api.spins && ed.ancillas === api.ancillas
            && ed.feasible === api.feasible
            && ed.constraints === (m.constraints || []).length
            && ed.terms === (m.objective?.terms || []).length;
  console.log(`  ${same ? "ok  " : "FAIL"} ${what}`);
  if (!same) {
    fail++;
    console.log(`       editor: ${ed.spins} spins, ${ed.ancillas} ancillas, feasible ${ed.feasible}, `
      + `${ed.constraints}/${(m.constraints || []).length} constraints, ${ed.terms}/${(m.objective?.terms || []).length} terms`);
    console.log(`       api   : ${api.spins} spins, ${api.ancillas} ancillas, feasible ${api.feasible}`);
  }
}

if (errs.length) { console.log("  page errors: " + errs.join(" | ")); fail++; }
await browser.close();
files.close();
if (SELFTEST) {
  const ok = fail > 0;
  console.log(ok ? "selftest ok: an editor that drops a k is caught"
                 : "SELFTEST FAILED: a damaged editor agreed with the API on every model");
  process.exit(ok ? 0 : 1);
}
console.log(fail ? `EDITOR vs API: ${fail} failed` : "EDITOR vs API ok");
process.exit(fail ? 1 : 0);
