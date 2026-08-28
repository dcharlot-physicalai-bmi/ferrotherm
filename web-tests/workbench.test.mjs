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

// --- encoding selection, and the caveat when it cannot be exact ---------------------------------
{
  // Ignoring "encoding" was a real bug on the HTTP surface: a document asking for binary got
  // one-hot with a different spin count and no error. Same document, same verdict, is the promise.
  const binary6 = JSON.stringify({
    variables: [{ name: "x", values: 6, encoding: "binary" }], tries: 4,
  });
  const r = await run(binary6);
  check("a binary encoding is honoured, not silently one-hot",
        /1 variables over 3 spins/.test(r.hint), r.hint);
  // Not `|| true`. A check that passes regardless of what the page did is worse than no check:
  // it reports green forever, including after the thing it names stops working.
  check("and the status says the encoding cannot be exact, over 'every constraint holds'",
        /caveat/i.test(r.status), r.status.slice(0, 90));

  const onehot6 = JSON.stringify({ variables: [{ name: "x", values: 6 }], tries: 4 });
  const o = await run(onehot6);
  check("one-hot is still the default, at k spins", /1 variables over 6 spins/.test(o.hint), o.hint);
  check("and an exact model reports no caveat, so the signal means something",
        !/caveat/i.test(o.status), o.status.slice(0, 70));

  const bad = await run(JSON.stringify({ variables: [{ name: "x", values: 6, encoding: "gray" }] }));
  check("an unknown encoding is refused by listing the ones that exist",
        /unknown encoding/.test(bad.status), bad.status.slice(0, 100));
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

// --- the optimality bracket -------------------------------------------------------------------
//
// The panel this suite most needs to cover, because its failure mode is silence: a stale bracket
// still RENDERS. Every check below is about the certificate matching the state beside it.
{
  const solve = (method) => page.evaluate((m) => {
    document.getElementById("method").value = m;
    document.getElementById("solve").click();
    const row = (n) => [...document.querySelectorAll("#btab tr")]
      .find(r => r.cells[0].textContent === n)?.cells[1].textContent ?? null;
    return {
      status: document.getElementById("status").textContent,
      verdict: document.getElementById("certverdict").textContent,
      hidden: document.getElementById("cert").hidden,
      energy: document.getElementById("r-e").textContent,
      best: document.querySelector("#btab tr.best")?.cells[0].textContent ?? null,
      sdp: row("sdp"),
      window: document.getElementById("brwindow").style.width,
      note: document.getElementById("certnote").textContent,
    };
  }, method);

  // A frustrated ring: 12 bonds, all but one satisfiable, and small enough to prove.
  const ring = { n: 12, couplings: [] };
  for (let i = 0; i < 12; i++) ring.couplings.push([i, (i + 1) % 12, i === 0 ? -1 : 1]);
  await run(JSON.stringify({ graph: ring, beta: 1, seed: 1 }));

  const br = await solve("branch");
  check("branch and bound proves a small instance", /tree exhausted/.test(br.status),
        br.status.slice(0, 100));
  check("and the panel says so", /PROVED OPTIMAL/.test(br.verdict), br.verdict);
  check("the bracket is shown once a solver has run", br.hidden === false);
  check("a frustrated 12-ring bottoms out at -10", /^-10/.test(br.energy), br.energy);
  check("a bound is named as the best one", br.best !== null, String(br.best));
  check("and the note separates the two proofs",
        /enumeration rather than by the bound|bound met the state/.test(br.note),
        br.note.slice(0, 110));

  const tb = await solve("tabu");
  check("tabu reports what it actually ran", /of 50,000 iterations/.test(tb.status),
        tb.status.slice(0, 90));
  check("and a search that is not a proof does not claim one", !/PROVED/.test(tb.verdict),
        tb.verdict);

  // The 2D lattice preset WRAPS, so it is a torus -- and the toroidal bound is by far the
  // strongest one available there. Checked on the preset rather than a hand-built graph, because
  // the row only appears when the page recognises the model as a torus.
  const torus = await page.evaluate(() => {
    document.getElementById("preset").value = "lattice";
    document.getElementById("preset").dispatchEvent(new Event("change"));
    document.getElementById("certify").click();
    const trs = [...document.querySelectorAll("#btab tr")];
    const val = (re) => {
      const r = trs.find(x => re.test(x.cells[0].textContent));
      return r ? parseFloat(r.cells[1].textContent) : NaN;
    };
    return {
      rows: trs.map(r => r.cells[0].textContent),
      best: document.querySelector("#btab tr.best")?.cells[0].textContent ?? null,
      torVal: val(/torus/),
      decVal: val(/decoupled/),
    };
  });
  check("the lattice preset is recognised as a torus", torus.rows.some(r => /torus/.test(r)),
        torus.rows.join(","));
  // It TIES rather than wins, and that is a fact about the instance: the preset is a FERROMAGNET,
  // where -sum|J| is not a relaxation at all but the ground energy itself. Asserting that the
  // toroidal bound is strictly best here would be asserting that the easy case does not occur.
  check("and it is attained, so it IS the maximum rather than a bound",
        torus.rows.some(r => /torus \(attained\)/.test(r)), torus.rows.join(","));
  check("and it ties the cheap bound a ferromagnet makes exact",
        Math.abs(torus.torVal - torus.decVal) < 1e-9,
        `torus ${torus.torVal} vs decoupled ${torus.decVal}`);

  const bl = await solve("bls");
  check("breakout local search reports its descents", /descents/.test(bl.status),
        bl.status.slice(0, 100));
  check("and does not claim a proof either", !/PROVED/.test(bl.verdict), bl.verdict);

  const pa = await solve("population");
  check("population annealing reports rho", /rho [\d.]+ of 512/.test(pa.status),
        pa.status.slice(0, 100));

  // THE ONE THAT MATTERS: a certificate must not outlive the model it is about.
  const stale = await page.evaluate(() => {
    const before = document.getElementById("cert").hidden;
    const el = document.getElementById("src");
    el.value = JSON.stringify({ graph: { builtin: "ring", n: 16, j: 1.0 }, beta: 1, seed: 0 });
    el.dispatchEvent(new Event("change", { bubbles: true }));
    return { before, after: document.getElementById("cert").hidden };
  });
  check("a new model clears the old certificate", stale.before === false && stale.after === true,
        JSON.stringify(stale));

  // And a sweep must retract the PROOF while leaving the bounds, which are about the graph.
  await run(JSON.stringify({ graph: ring, beta: 1, seed: 1 }));
  const after = await page.evaluate(() => {
    document.getElementById("method").value = "branch";
    document.getElementById("solve").click();
    const proved = document.getElementById("certverdict").textContent;
    document.getElementById("run").click();   // start sweeping
    return new Promise(res => setTimeout(() => {
      document.getElementById("run").click(); // stop
      res({ proved, now: document.getElementById("certverdict").textContent,
            shown: document.getElementById("cert").hidden === false });
    }, 250));
  });
  check("sweeping retracts the proof", /PROVED/.test(after.proved) && !/PROVED/.test(after.now),
        after.proved.slice(0, 40) + " -> " + after.now.slice(0, 40));
  check("but the bounds survive, because they are about the graph", after.shown);
}

check("no page errors", errs.length === 0, errs.join(" | "));

await browser.close();
server.close();
console.log(failed ? `\n${failed} failed` : "\nall passed");
process.exit(failed ? 1 : 0);
