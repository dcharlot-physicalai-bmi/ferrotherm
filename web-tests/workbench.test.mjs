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

const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu"] });
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

// --- a model arriving from the node editor ---------------------------------------------------------
//
// "Open in workbench" in graph.html is a link into this page carrying the model in the fragment.
// Both pages take the same JSON -- the Model pane says so -- so the handoff is a link rather than a
// translation, and this is the assertion that it stays one.
{
  const MODEL = {
    variables: [{ name: "west", values: 3 }, { name: "middle", values: 3 }, { name: "east", values: 3 }],
    constraints: [{ type: "all_different", of: [{ var: "west" }, { var: "middle" }, { var: "east" }] }],
  };
  const b64 = Buffer.from(JSON.stringify(MODEL)).toString("base64url");

  const linked = await browser.newPage();
  const linkedErrs = [];
  linked.on("pageerror", e => linkedErrs.push(e.message));
  await linked.goto(`${base}/ide.html#model=${b64}`);
  await linked.waitForFunction(() => document.getElementById("wasminfo").textContent.includes("loaded"));

  const seeded = await linked.evaluate(() => {
    try { return JSON.parse(document.getElementById("src").value); } catch { return null; }
  });
  check("a linked model seeds the workbench, not the default lattice",
        seeded?.variables?.length === 3 && !seeded.graph,
        JSON.stringify(seeded?.variables?.map(v => v.name) ?? seeded));

  // A `variables` spec is SOLVED as it is applied -- a problem has no lattice, so the answer is
  // drawn where the lattice view would be, and #viewhint is the readable trace of that.
  const hint = await linked.evaluate(() => document.getElementById("viewhint").textContent);
  check("and the workbench solves it on arrival", /3 variables over \d+ spins/.test(hint),
        hint.slice(0, 70));

  // A fragment pasted into an already-open workbench: only the hash changes, so the browser does
  // not reload and the page has to look for itself.
  const second = Buffer.from(JSON.stringify({
    variables: [{ name: "solo", values: 4 }, { name: "duo", values: 4 }],
    constraints: [{ type: "not_equal", a: "solo", b: "duo" }],
  })).toString("base64url");
  await linked.evaluate(h => { location.hash = h; }, `#model=${second}`);
  const swapped = await linked.evaluate(() => JSON.parse(document.getElementById("src").value));
  check("a link pasted into an open workbench is followed",
        swapped.variables.length === 2 && swapped.variables[0].name === "solo",
        JSON.stringify(swapped.variables.map(v => v.name)));

  // A fragment that is not a model must not take the page down with it.
  await linked.goto(`${base}/ide.html#model=not-base64-at-all!!`);
  await linked.reload();
  await linked.waitForFunction(() => document.getElementById("wasminfo").textContent.includes("loaded"));
  const fell = await linked.evaluate(() => JSON.parse(document.getElementById("src").value));
  check("a fragment that does not decode falls back to the preset", !!fell.graph,
        "not an error page");
  check("and no page error escapes the linked route", linkedErrs.length === 0, linkedErrs.join(" | "));
  await linked.close();
}

// --- a higher-order model, solved natively in the browser ------------------------------------------
//
// The third request shape. A term over three or more variables in a "variables" spec goes through
// the reduction -- one ancilla per substituted pair, plus a penalty ~1300 against term weights of
// 1 that makes the landscape rigid. This path spends none of that, and the assertion below is the
// one that separates them: the ancilla count the OTHER path would have paid.
{
  const run = spec => page.evaluate(s => {
    document.getElementById("src").value = JSON.stringify(s);
    document.getElementById("src").dispatchEvent(new Event("change"));
    return {
      hint: document.getElementById("viewhint").textContent,
      status: document.getElementById("status").textContent,
      curl: document.getElementById("curl").textContent,
      mcp: document.getElementById("mcpcall").textContent,
      ftp: document.getElementById("ftp").textContent,
    };
  }, spec);

  const parity = await run({ spins: 3, terms: [{ vars: [0, 1, 2], weight: 1.0 }], seed: 7 });
  check("a three-body model runs in the browser", /3 spins, 1 terms of arity up to 3/.test(parity.hint),
        parity.hint);
  check("and says it spent no ancillas", /no ancillas/.test(parity.status), parity.status.slice(0, 70));
  check("the curl and MCP panes name the right operation",
        /v1\/hubo/.test(parity.curl) && /ferrotherm_hubo/.test(parity.mcp));
  // A higher-order model has no pairwise program, and saying so is better than showing an empty
  // pane that reads as a bug.
  check("and the .ftp pane explains why there is none", /is pairwise, and this model is not/.test(parity.ftp),
        parity.ftp.split("\n")[0]);

  // The refusal path, by name: s * s = 1, so a repeat is a different term than the one written.
  const repeat = await run({ spins: 3, terms: [{ vars: [0, 0, 1], weight: 1.0 }] });
  check("a repeated variable is refused by name",
        /already in this term/.test(repeat.status), repeat.status.slice(0, 80));

  const outside = await run({ spins: 3, terms: [{ vars: [0, 9], weight: 1.0 }] });
  check("and so is one out of range", /no variable 9/.test(outside.status), outside.status.slice(0, 60));

  // Two shapes at once is two operations, not a merge.
  const both = await page.evaluate(() => {
    document.getElementById("src").value =
      JSON.stringify({ graph: { builtin: "lattice2d", l: 4, j: 1 }, terms: [] });
    document.getElementById("src").dispatchEvent(new Event("change"));
    return document.getElementById("status").textContent;
  });
  check("two request shapes at once is refused", /not 2|different operations/.test(both),
        both.slice(0, 70));

  const preset = await page.evaluate(() => {
    const s = document.getElementById("preset");
    s.value = "hubo";
    s.dispatchEvent(new Event("change"));
    return {
      hint: document.getElementById("viewhint").textContent,
      beta: document.getElementById("betaval").textContent,
    };
  });
  check("the shipped higher-order preset runs", /8 spins, 6 terms of arity up to 4/.test(preset.hint),
        preset.hint);
  // A model with no single beta must not blank the readout by setting the slider from undefined.
  check("and a spec with no beta leaves the slider readable", /^\d/.test(preset.beta.trim()),
        JSON.stringify(preset.beta));
}

// --- the live view must not be frame-bound ---------------------------------------------------------
//
// It ran ONE sweep per requestAnimationFrame, which pinned it near 60 sweeps a second on every
// model and every machine: a 5-ring and a 512x512 lattice sampled at exactly the same rate, and
// making the sampler faster moved that number by nothing. The cap was the frame, not the physics.
//
// The threshold below is deliberately far above 60 rather than near whatever this machine happens
// to manage. A rate assertion tuned to one machine fails on a slower one and says nothing on a
// faster one; this asks only whether the frame is still the ceiling, which is a structural question
// with a structural answer.
{
  const rate = await page.evaluate(async () => {
    const sel = document.getElementById("preset");
    sel.value = "frustrated";
    sel.dispatchEvent(new Event("change"));
    await new Promise(r => setTimeout(r, 150));
    const read = () => +document.getElementById("r-s").textContent.replace(/,/g, "");
    const before = read();
    document.getElementById("run").click();
    await new Promise(r => setTimeout(r, 1200));
    document.getElementById("run").click();
    return read() - before;
  });
  check("the live view is not pinned to one sweep per frame", rate > 5000,
        `${rate.toLocaleString()} sweeps in ~1.2s; one-per-frame would be about 72`);
}

// --- the GPU path must not be paying for round trips -----------------------------------------------
//
// docs/ide.html used to build a params buffer, a bind group, an encoder and a SUBMIT per
// (sweep x colour class): 400 driver round trips for the 200-sweep panel, plus a fresh
// requestDevice and shader compile on every call. gpu/src/lib.rs records the same bug natively and
// what it looked like -- "~60 ms almost independent of node count" -- and names the tell:
//
//     Constant time under a growing workload is the signature of paying for round trips
//     rather than arithmetic.
//
// So this asserts the SHAPE rather than a rate: per-node cost must FALL as the model grows. A
// wall-clock threshold would be a statement about whatever machine ran it; this is a statement
// about where the time goes, and it holds on a slow machine and a fast one alike.
//
// Skipped where WebGPU is absent, because a headless runner without it can say nothing here.
{
  const hasGpu = await page.evaluate(async () => {
    if (!navigator.gpu) return false;
    try { return !!(await navigator.gpu.requestAdapter()); } catch { return false; }
  });
  if (!hasGpu) {
    console.log("SKIP  the GPU path shape   no WebGPU adapter in this browser");
  } else {
    const pts = [];
    for (const l of [32, 128]) {
      pts.push(await page.evaluate(async l => {
        document.getElementById("src").value =
          JSON.stringify({ graph: { builtin: "lattice2d", l, j: 1.0 }, beta: 0.44, seed: 1 });
        document.getElementById("src").dispatchEvent(new Event("change"));
        await new Promise(r => setTimeout(r, 250));
        const g = await gpuSweep(200);
        return { n: W.ft_len(sim), ms: g.ms };
      }, l));
    }
    const per = pts.map(p => p.ms / p.n);
    check("the GPU path is arithmetic-bound, not round-trip-bound", per[1] < per[0] * 0.9,
          `${(per[0] * 1000).toFixed(2)} us/node at n=${pts[0].n} -> ${(per[1] * 1000).toFixed(2)} at n=${pts[1].n}`);
    check("and the spins actually moved", pts.every(p => p.ms > 0),
          "a silently-failed dynamic offset leaves the state frozen and the timing near zero");
  }
}

// --- the sampler returns SAMPLES, and the page says what may be computed from them ----------------
//
// Every other control on this page hands back one state. The readouts beside the canvas are the
// order parameter of the single configuration the machine is sitting in -- a draw from a
// distribution, not an estimate of it. This checks that the Draw panel exists, that it fills, and
// that the interval it quotes is the corrected one rather than sigma/sqrt(N), which is the whole
// reason the panel is worth having.
{
  const r = await page.evaluate(async () => {
    // Critical, deliberately. At beta_c a 2D lattice shows critical slowing down, so tau is well
    // above its floor and the autocorrelation correction has something to correct.
    document.getElementById("src").value =
      JSON.stringify({ graph: { builtin: "lattice2d", l: 16, j: 1.0 }, beta: 0.44, seed: 3 });
    document.getElementById("src").dispatchEvent(new Event("change"));
    await new Promise(r => setTimeout(r, 200));
    const hiddenBefore = document.getElementById("samp").hidden;
    document.getElementById("draws").click();
    await new Promise(r => setTimeout(r, 400));
    const txt = id => document.getElementById(id).textContent;
    return {
      hiddenBefore,
      hiddenAfter: document.getElementById("samp").hidden,
      draws: txt("s-draws"), distinct: txt("s-distinct"), best: txt("s-best"),
      mag: txt("s-mag"), energy: txt("s-energy"), ess: txt("s-ess"), tau: txt("s-tau"),
      note: txt("sampnote"), status: txt("status"),
      beta: +document.getElementById("beta").value / 100,
      // The two intervals, read straight off the module, so the assertion is about arithmetic and
      // not about a rendered string.
      raw: (() => {
        const p = W.ft_scratch(32);
        if (!W.ft_samples_magnetization(sim, p)) return null;
        const dv = new DataView(W.memory.buffer, p, 32);
        return { value: dv.getFloat64(0, true), stderr: dv.getFloat64(8, true),
                 ess: dv.getFloat64(16, true), tau: dv.getFloat64(24, true),
                 n: W.ft_samples_len(sim) };
      })(),
    };
  });

  check("the panel is hidden until something is drawn", r.hiddenBefore === true);
  check("and appears once it is", r.hiddenAfter === false);
  // The body said 0.44 and the page must have sampled at 0.44. It used to read the slider and
  // ignore the body: this same block found it holding 1.50 from a preset loaded steps earlier.
  check("the body's beta is the beta it sampled at", Math.abs(r.beta - 0.44) < 1e-9, String(r.beta));
  check("it reports how many draws it kept", /^2,?000$/.test(r.draws), r.draws);
  check("and how many of them were distinct",
        +r.distinct.replace(/,/g, "") > 1 && +r.distinct.replace(/,/g, "") <= 2000, r.distinct);
  check("every tile is filled", ![r.best, r.mag, r.energy, r.ess, r.tau].includes("\u2014"),
        [r.best, r.mag, r.energy, r.ess, r.tau].join(" | "));
  check("<M> and <E> both carry an interval", /\u00b1/.test(r.mag) && /\u00b1/.test(r.energy),
        r.mag + "   " + r.energy);

  // THE CLAIM THE PANEL MAKES, in two parts.
  //
  // First the arithmetic, which holds always: ess is n/(2*tau) and nothing else. This is the
  // assertion that catches the page quietly reverting to sigma/sqrt(N), because that would need
  // ess to equal n at a tau above the floor.
  check("the effective sample size is the draw count deflated by tau",
        r.raw && Math.abs(r.raw.ess - r.raw.n / (2 * r.raw.tau)) < 1e-6,
        r.raw ? `ess ${r.raw.ess.toFixed(1)} = ${r.raw.n} / (2 x ${r.raw.tau.toFixed(3)})` : "no estimate");

  // Then that the correction BITES on this model. It is not a universal claim and must not be
  // written as one: tau bottoms out at 0.5, where n/(2*tau) is exactly n, and a genuinely
  // fast-mixing observable legitimately reports ess = draws. An earlier version of this test
  // asserted `ess < draws` unconditionally and failed against a correct page -- on an ORDERED
  // lattice whose chain had frozen into two states, where the energy trace really does decorrelate
  // in half a sweep because nothing is moving. That is what the distinct count below is for.
  check("and at criticality it is materially below the draw count",
        r.raw && r.raw.tau > 1.0 && r.raw.ess < 0.5 * r.raw.n,
        r.raw ? `tau ${r.raw.tau.toFixed(2)}, ess ${r.raw.ess.toFixed(0)} of ${r.raw.n}` : "no estimate");
  check("and the note says which interval it is",
        /sqrt\(var \/ ess\)/.test(r.note) && /NOT sqrt\(var \/ draws\)/.test(r.note),
        r.note.slice(0, 90));

  // A misaligned f64 write is undefined behaviour that works everywhere it is tried. `ft_scratch`
  // returns the buffer of a Vec<u8>, aligned to one, so this is the surface that would find it.
  check("the estimate survives an unaligned scratch pointer",
        r.raw && Number.isFinite(r.raw.value) && Math.abs(r.raw.value) <= 1,
        r.raw ? String(r.raw.value) : "no estimate");

  // AND THE READING THE DRAW COUNT ALONE WOULD NOT GIVE. An ordered lattice's chain freezes into a
  // handful of configurations while its energy still jitters, so tau reports fast mixing for a
  // machine that has not moved. The distinct count is what says so, and the panel must say it in
  // words rather than leaving a reader to compare two numbers.
  const frozen = await page.evaluate(async () => {
    document.getElementById("src").value =
      JSON.stringify({ graph: { builtin: "lattice2d", l: 8, j: 1.0 }, beta: 1.5, seed: 3 });
    document.getElementById("src").dispatchEvent(new Event("change"));
    await new Promise(r => setTimeout(r, 200));
    document.getElementById("draws").click();
    await new Promise(r => setTimeout(r, 400));
    const txt = id => document.getElementById(id).textContent;
    return { distinct: +txt("s-distinct").replace(/,/g, ""), tau: txt("s-tau"),
             mag: txt("s-mag"), ess: txt("s-ess"), note: txt("sampnote") };
  });
  check("a frozen chain returns one state two thousand times", frozen.distinct === 1,
        `${frozen.distinct} distinct`);
  check("and the panel says so in words", /explored almost nothing/.test(frozen.note));
  // The arithmetic here is right and reads as certainty: one repeated state has zero sample
  // variance, so the interval is a point. The infinite tau beside it is the only thing standing
  // between a reader and that reading, and it must be on the panel rather than in a comment.
  check("its interval is zero-width, and the tau beside it says why",
        /\u00b1 0\.0*$/.test(frozen.mag.trim()) && frozen.tau === "infinite",
        `${frozen.mag}   tau ${frozen.tau}, ess ${frozen.ess}`);

  // Drawn states belong to the model they came from.
  const cleared = await page.evaluate(async () => {
    document.getElementById("src").value =
      JSON.stringify({ graph: { builtin: "ring", n: 12, j: 1.0 }, beta: 0.5, seed: 1 });
    document.getElementById("src").dispatchEvent(new Event("change"));
    await new Promise(r => setTimeout(r, 200));
    return document.getElementById("samp").hidden;
  });
  check("and a new model clears them rather than showing them beside it", cleared === true);
}

check("no page errors", errs.length === 0, errs.join(" | "));

await browser.close();
server.close();
console.log(failed ? `\n${failed} failed` : "\nall passed");
process.exit(failed ? 1 : 0);
