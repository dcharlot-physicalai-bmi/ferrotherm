// The workbench, driven in a real browser.
//
// Its pane says "the same JSON the API and the MCP tools take". That was true of `sample` and
// `anneal` and false of `solve` -- the operation the MCP server tells an agent to reach for first.
// These tests check the claim rather than the code: a request body pasted from the tool
// description has to run here unchanged.

import { chromium } from "playwright";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join } from "node:path";

const ROOT = new URL("../docs/", import.meta.url).pathname;
const MIME = { ".html": "text/html", ".wasm": "application/wasm", ".js": "text/javascript" };

const server = createServer(async (req, res) => {
  const p = join(ROOT, (req.url === "/" ? "/ide.html" : req.url).split("?")[0]);
  try {
    const body = await readFile(p);
    res.writeHead(200, { "content-type": MIME[extname(p)] ?? "application/octet-stream" });
    res.end(body);
  } catch {
    res.writeHead(404).end("not found");
  }
});
await new Promise(r => server.listen(0, r));
const base = `http://localhost:${server.address().port}`;

let failed = 0;
const check = (name, ok, detail = "") => {
  console.log(`${ok ? "PASS" : "FAIL"} ${name}${detail ? "   " + detail : ""}`);
  if (!ok) failed++;
};

const browser = await chromium.launch();
const page = await browser.newPage();
const errs = [];
page.on("pageerror", e => errs.push("pageerror: " + e.message));
page.on("console", m => { if (m.type() === "error") errs.push("console: " + m.text()); });

await page.goto(base + "/ide.html");
await page.waitForFunction(() => document.getElementById("status")?.textContent
  && !document.getElementById("status").textContent.includes("loading"), null, { timeout: 20000 });

/** Paste a request body into the pane and apply it, returning what the page shows. */
const run = (body) => page.evaluate((json) => {
  const el = document.getElementById("src");
  el.value = json;
  el.dispatchEvent(new Event("input", { bubbles: true }));
  el.dispatchEvent(new Event("change", { bubbles: true }));
  // whatever the page binds to, force the rebuild the same way the UI does
  if (typeof window.apply === "function") window.apply();
  else document.getElementById("apply")?.click();
  return {
    status: document.getElementById("status").textContent,
    hint: document.getElementById("viewhint").textContent,
    curl: document.getElementById("curl").textContent,
    mcp: document.getElementById("mcpcall").textContent,
    ftp: document.getElementById("ftp").dataset.full ?? document.getElementById("ftp").textContent,
  };
}, body);

// --- a spin model still works, unchanged ------------------------------------------------------
{
  const r = await run(JSON.stringify({ graph: { builtin: "ring", n: 8, j: 1.0 }, beta: 1, seed: 0 }));
  check("a graph still samples", /8 nodes/.test(r.hint) || /model built/.test(r.status),
        r.status.slice(0, 70));
  check("and exports a sample request", r.curl.includes("/v1/sample"));
}

// --- a problem, which is the thing that did not work --------------------------------------------
{
  const colouring = JSON.stringify({
    variables: [{ name: "west", values: 3 }, { name: "middle", values: 3 }, { name: "east", values: 3 }],
    constraints: [
      { type: "not_equal", a: "west", b: "middle" },
      { type: "not_equal", a: "middle", b: "east" },
      { type: "not_equal", a: "west", b: "east" },
    ],
    tries: 12,
  });
  const r = await run(colouring);
  check("a problem solves", /every constraint holds/.test(r.status), r.status.slice(0, 70));
  check("and reports variables over spins", /3 variables over 9 spins/.test(r.hint), r.hint);
  check("it exports a SOLVE request, not a sample one",
        r.curl.includes("/v1/solve") && r.mcp.includes("ferrotherm_solve"));
  check("and the compiled program comes with it", r.ftp.startsWith("ftp 1"));
}

// --- all_different: the constraint no pair of variables can express ----------------------------
{
  const latin = JSON.stringify({
    variables: [0,1,2,3].map(i => ({ name: `c${i}`, values: 4 })),
    constraints: [{ type: "all_different", of: [0,1,2,3].map(i => ({ var: `c${i}` })) }],
    tries: 60,
  });
  const r = await run(latin);
  check("all_different solves a latin square row", /every constraint holds/.test(r.status), r.status.slice(0, 80));

  // Five variables over three values has no answer at any penalty. Annealing it and reporting
  // infeasible reads as "raise the penalty", which is advice that cannot work here.
  const pigeon = JSON.stringify({
    variables: [0,1,2,3,4].map(i => ({ name: `x${i}`, values: 3 })),
    constraints: [{ type: "all_different", of: [0,1,2,3,4].map(i => ({ var: `x${i}` })) }],
  });
  const p2 = await run(pigeon);
  check("and refuses the pigeonhole case by name rather than annealing it",
        /No assignment can satisfy|pigeonhole/.test(p2.status), p2.status.slice(0, 110));
}

// --- the GPU readback guard, which nothing reached ----------------------------------------------
{
  // `ft_set_spins` validates length and +/-1-ness, was written and unit-tested, and was called by
  // NOTHING: the page wrote GPU results straight into wasm memory after coercing every value with
  // `> 0 ? 1 : -1`, which launders a dropped dispatch or a short readback into a plausible state
  // that is then scored with confidence. A headless browser has no adapter, so the synthetic cases
  // below are what can be checked -- and they are the ones that were never checked.
  await run(JSON.stringify({ graph: { builtin: "lattice2d", l: 4, j: 1.0 }, beta: 0.44, seed: 1 }));

  const probe = (state) => page.evaluate((s) => {
    try { return { ok: true, energy: window.__putSpins(Int32Array.from(s)) }; }
    catch (e) { return { ok: false, message: e.message }; }
  }, state);

  const n = 16;
  // All +1 on a ferromagnetic lattice is the ground state, so this asserts the state ARRIVED: a
  // no-op that left the previous state in place would not land on -32, and a stripe pattern -- the
  // first thing tried here -- scores exactly 0, which a no-op could produce too.
  const good = await probe(Array.from({ length: n }, () => 1));
  check("a valid readback is accepted and lands, scoring the ferromagnetic ground state",
        good.ok && good.energy === -32, JSON.stringify(good));

  const short = await probe(Array.from({ length: n - 3 }, () => 1));
  check("a readback shorter than the model is REFUSED, not padded",
        !short.ok && /did not complete/.test(short.message), JSON.stringify(short));

  const junk = await probe(Array.from({ length: n }, (_, i) => (i === 5 ? 0 : 1)));
  check("a value that is not a spin is refused rather than coerced",
        !junk.ok && /\+1\/-1/.test(junk.message), JSON.stringify(junk));
}

// --- a soft constraint, which is a different ANSWER rather than a different number ---------------
{
  // A preference the solver may trade away. The failure this guards is not an exception: it is the
  // page reporting "every constraint holds" over an answer that broke one, which nothing else on
  // the screen contradicts. The status line has to distinguish a rule from a price.
  const body = (extra) => JSON.stringify({
    variables: [{ name: "a", values: 2 }, { name: "b", values: 2 }],
    constraints: [{ type: "not_equal", a: "a", b: "b", ...extra }],
    objective: { maximize: true, terms: [
      { var: "a", value: 0, weight: 5 }, { var: "b", value: 0, weight: 5 }] },
    tries: 24,
  });

  const cheap = await run(body({ soft: 1 }));
  check("a cheap preference is traded, and the status says so, not \"every constraint holds\"",
        /1 preference\(s\) traded for 1\.00/.test(cheap.status), cheap.status.slice(0, 90));
  check("and the answer is still feasible, because a traded preference is not a broken rule",
        !/broke/.test(cheap.status), cheap.status.slice(0, 90));

  const dear = await run(body({ soft: 50 }));
  check("priced above the objective the same preference is KEPT",
        /every constraint holds/.test(dear.status), dear.status.slice(0, 90));

  const hard = await run(body({}));
  check("and with no price at all it is a rule, kept as before",
        /every constraint holds/.test(hard.status), hard.status.slice(0, 90));

  const bad = await run(body({ soft: "5" }));
  check("a price that is not a number is refused by name, not read as a hard constraint",
        /soft/.test(bad.status), bad.status.slice(0, 110));
}

// --- an inequality, with the answer visible ------------------------------------------------------
{
  const r = await run(JSON.stringify({
    variables: Array.from({ length: 5 }, (_, i) => ({ name: `d${i}`, values: 2 })),
    constraints: [{ type: "at_most", k: 2, of: Array.from({ length: 5 }, (_, i) => ({ var: `d${i}`, value: 1 })) }],
    objective: { maximize: true, terms: Array.from({ length: 5 }, (_, i) => ({ var: `d${i}`, value: 1, weight: 5 - i })) },
    tries: 16,
  }));
  check("an inequality binds", /every constraint holds/.test(r.status), r.status.slice(0, 70));
  check("and costs slack spins", /over 1[0-9] spins/.test(r.hint), r.hint);
}

// --- an infeasible answer explains itself --------------------------------------------------------
{
  const r = await run(JSON.stringify({
    variables: [{ name: "a", values: 3 }, { name: "b", values: 3 }],
    constraints: [{ type: "not_equal", a: "a", b: "b" }],
    objective: { maximize: true, terms: [{ var: "a", value: 1, weight: 40 }, { var: "b", value: 1, weight: 40 }] },
    penalty: 1, tries: 12,
  }));
  check("a broken constraint is reported, not hidden", /constraint\(s\) broke/.test(r.status),
        r.status.slice(0, 80));
}

// --- bad input teaches ---------------------------------------------------------------------------
{
  const both = await run(JSON.stringify({ graph: { builtin: "ring", n: 4 }, variables: [] }));
  check("graph and variables together is refused",
        /not both|different operations/.test(both.status), both.status.slice(0, 80));

  const neither = await run(JSON.stringify({ beta: 1 }));
  check("and neither names both options",
        /"graph"/.test(neither.status) && /"variables"/.test(neither.status),
        neither.status.slice(0, 90));

  const badValue = await run(JSON.stringify({
    variables: [{ name: "t", lo: 10, hi: 20 }],
    constraints: [{ type: "fix", var: "t", value: 3 }],
  }));
  check("an out-of-range value names the range", /10\.\.=20/.test(badValue.status),
        badValue.status.slice(0, 90));

  const unknown = await run(JSON.stringify({
    variables: [{ name: "a", values: 3 }, { name: "b", values: 3 }],
    constraints: [{ type: "nonsense", a: "a", b: "b" }],
  }));
  check("an unknown constraint lists the known ones", /at_most_one/.test(unknown.status),
        unknown.status.slice(0, 90));
}

check("no page errors", errs.length === 0, errs.join(" | "));

await browser.close();
server.close();
console.log(failed ? `\n${failed} failed` : "\nall passed");
process.exit(failed ? 1 : 0);
