// Editor tests, driven through the same scriptable API an agent would use.
//
// Every assertion here corresponds to something that can break without anyone noticing: a port
// that stops accepting a link, a constraint that compiles to the wrong penalty, a saved document
// that no longer loads. The browser reports none of those as a failure on its own.

import { chromium } from "playwright";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join } from "node:path";

const ROOT = new URL("../docs/", import.meta.url).pathname;
const MIME = { ".html": "text/html", ".wasm": "application/wasm", ".js": "text/javascript" };

const server = createServer(async (req, res) => {
  const p = join(ROOT, (req.url === "/" ? "/graph.html" : req.url).split("?")[0]);
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

await page.goto(base + "/graph.html");
await page.evaluate(() => window.ferrotherm.ready);

// --- the example that ships with the page must actually run ---------------------------------------
{
  const txt = await page.evaluate(() => window.ferrotherm.run());
  const colours = [...txt.matchAll(/(west|middle|east) = (\d)/g)].map(m => m[2]);
  check("the shipped example solves", new Set(colours).size === 3 && colours.length === 3,
        `three regions got ${JSON.stringify(colours)}`);
  check("variables answer by name", txt.includes("west ="), "not by node number");
}

// --- how many ways are there to do the job ---------------------------------------------------------
//
// The editor could answer "what should I do" and not "was that the only way". A model with a
// symmetry has several optima and the solve threw all but one away; a modeller reading a single
// assignment had no way to tell a unique answer from one of several. Exactly-one over three
// binaries has three, and the count is known in advance rather than observed.
//
// PLACED AFTER THE SHIPPED-EXAMPLE CHECKS ON PURPOSE: this block clears the canvas to build its own
// model, and running it first left the next test looking at an empty graph and reporting the page
// as broken.
{
  const txt = await page.evaluate(() => {
    const F = window.ferrotherm;
    for (const n of [...F.nodes]) F.remove(n.id);
    const a = F.add("binary", 40, 40, { name: "a" });
    const b = F.add("binary", 40, 120, { name: "b" });
    const c = F.add("binary", 40, 200, { name: "c" });
    const one = F.add("exactlyone", 240, 120);
    for (const v of [a, b, c]) F.connect(v, one);
    const solve = F.add("solve", 440, 120, { tries: 40 });
    F.connect(one, solve);
    const rep = F.add("report", 640, 120);
    F.connect(solve, rep, "result");
    return F.run();
  });

  check("the editor counts the ways to do the job", /3 distinct ways to do this/.test(txt),
        txt.split("\n").filter(l => /distinct ways|only one way/.test(l))[0] || txt.slice(-160));

  // Each listed alternative must be a DIFFERENT assignment with exactly one variable set. A block
  // that printed the same answer three times would satisfy a count check and nothing else.
  const rows = [...txt.matchAll(/^\s+\d+\.\s+(a=\d\s+b=\d\s+c=\d)$/gm)].map(m => m[1].replace(/\s+/g, " "));
  check("and lists three different ones", new Set(rows).size === 3, JSON.stringify(rows));
  check("each sets exactly one variable",
        rows.length === 3 && rows.every(r => (r.match(/=1/g) || []).length === 1), JSON.stringify(rows));

  // A count without this sentence reads as a census, and it is not one.
  check("and says it is evidence rather than a proof", /not a\s*\n?\s*proof there are no others/.test(txt),
        (txt.match(/found by \d+ independent tries[^)]*/) || ["no caveat printed"])[0].replace(/\s+/g, " "));

  // Everything above the block was read off the same handle, so selecting an optimum must put it
  // back. If it did not, the printed answer and the listed alternatives would disagree.
  const head = txt.split("distinct ways")[0];
  check("the printed answer is still the one solve returned",
        /\ba = \d/.test(head) && !/did not decode/.test(head), head.slice(-120).replace(/\n/g, " | "));
}


// --- the three counting constraints differ in the way they are supposed to -------------------------
//
// All three are "k of these"; only the comparison changes. Rewarding every variable unequally makes
// the difference visible: a ceiling has to hold against the reward, a floor has to hold against its
// absence, and an equality has to do both.
const counting = (kind, k, reward = "up") => page.evaluate(([kind, k, reward]) => {
  const F = window.ferrotherm;
  F.clear();
  const vs = [];
  for (let i = 0; i < 4; i++) vs.push(F.add("binary", 40, 40 + i * 90, { name: "v" + i }));
  const c = F.add(kind, 320, 60, { k, value: 1 });
  ["a", "b", "c", "d"].forEach((pin, i) => F.connect(vs[i], c, pin));
  const s = F.add("solve", 620, 200);
  F.connect(c, s);
  vs.forEach((v, i) => {
    const o = F.add("prefer", 320, 360 + i * 80,
                    { value: 1, weight: reward === "up" ? 4 - i : -(4 - i), maximize: 1 });
    F.connect(v, o, "var");
    F.connect(o, s);
  });
  F.connect(s, F.add("report", 900, 200), "result");
  const txt = F.run();
  return {
    on: [...txt.matchAll(/v(\d) = 1/g)].map(m => "v" + m[1]),
    terms: +txt.match(/(\d+) objective terms/)[1],
    spins: +txt.match(/(\d+) spins/)[1],
    feasible: txt.includes("feasible  yes"),
  };
}, [kind, k, reward]);

{
  const r = await counting("atmost", 2);
  check("at most k binds as a ceiling",
        r.feasible && r.on.length === 2 && r.on.join() === "v0,v1",
        `took ${JSON.stringify(r.on)} of four, each rewarded`);

  const least = await counting("atleast", 3);
  check("at least k is only a floor",
        least.feasible && least.on.length === 4,
        `took all four, since a floor does not forbid more`);

  const exact = await counting("cardinality", 2);
  check("exactly k is both at once", exact.feasible && exact.on.length === 2);

  // The assertion that actually separates an inequality from an equality. With the reward pushing
  // the other way, "at most two" is satisfied by taking none and "exactly two" is not. Without
  // this, at_most compiled as cardinality passes every other check in this file.
  const slack = await counting("atmost", 2, "down");
  const forced = await counting("cardinality", 2, "down");
  check("at most k is satisfied by taking none",
        slack.feasible && slack.on.length === 0,
        `took ${JSON.stringify(slack.on)} when taking any was penalised`);
  check("where exactly k is not", forced.feasible && forced.on.length === 2,
        `an equality still has to take ${forced.on.length}`);

  check("an inequality costs slack spins", r.spins > exact.spins,
        `${r.spins} against ${exact.spins} for the exact form`);
  check("the slack never appears in the answer", r.on.every(v => v.startsWith("v")));
  check("every objective term compiles", r.terms === 4,
        "a fixed number of Solve ports would have dropped three");
}

// --- a bad annealing ladder is refused in words, not silently defaulted ----------------------------
{
  const msg = await page.evaluate(() => {
    const F = window.ferrotherm;
    F.set(F.nodes.find(n => n.type === "solve").id, "beta cold", 0.001);
    return F.run();
  });
  check("a backwards ladder is refused", msg.includes("beta cold must exceed beta hot"),
        "and says which way round it goes");
}

// --- documents written before Solve grew its ports still load --------------------------------------
{
  const txt = await page.evaluate(() => {
    const doc = {
      format: "ferrotherm-graph/1",
      nodes: [
        { id: 1, type: "categorical", x: 40, y: 40, values: { name: "a", values: 3 } },
        { id: 2, type: "categorical", x: 40, y: 160, values: { name: "b", values: 3 } },
        { id: 3, type: "notequal", x: 280, y: 60, values: {} },
        { id: 4, type: "solve", x: 520, y: 60, values: { tries: 8 } },
        { id: 5, type: "report", x: 760, y: 60, values: {} },
      ],
      links: [
        { from: 1, to: 3, port: "a", kind: "var" }, { from: 2, to: 3, port: "b", kind: "var" },
        { from: 3, to: 4, port: "model", kind: "cons" },
        { from: 4, to: 5, port: "result", kind: "result" },
      ],
    };
    window.ferrotherm.fromJson(JSON.stringify(doc));
    return window.ferrotherm.run();
  });
  check("a document naming the old Solve ports still loads and runs",
        txt.includes("feasible  yes") && /a = \d/.test(txt));
}

// --- save and load round-trip -----------------------------------------------------------------------
{
  const same = await page.evaluate(() => {
    const F = window.ferrotherm;
    const before = F.toJson();
    F.fromJson(before);
    return before === F.toJson();
  });
  check("a document survives a save and load round trip", same);
}

// --- search finds what is already on the canvas, not only what could be added -----------------------
{
  const r = await page.evaluate(() => {
    const F = window.ferrotherm;
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "k", ctrlKey: true, bubbles: true }));
    const q = document.getElementById("palq");
    q.value = "a"; q.dispatchEvent(new Event("input"));
    const rows = [...document.getElementById("palr").children].map(d => d.textContent);
    return { rows, find: F.find("a") };
  });
  check("search reaches existing nodes", r.rows.some(t => t.startsWith("→")),
        r.rows.find(t => t.startsWith("→")) ?? "no travel entries offered");
  check("and can still add new ones", r.rows.some(t => t.startsWith("+")));
  check("find() agrees with the palette", r.find.length > 0);
}

// --- the API refuses what the mouse would refuse -----------------------------------------------------
{
  const errors = await page.evaluate(() => {
    const F = window.ferrotherm;
    const out = [];
    const grab = fn => { try { fn(); out.push(null); } catch (e) { out.push(e.message); } };
    const v = F.add("binary", 0, 0), r = F.add("report", 0, 0);
    grab(() => F.connect(v, r, "result"));      // a variable is not a result
    grab(() => F.connect(v, r, "nonesuch"));    // no such port
    grab(() => F.add("teleporter"));            // no such type
    grab(() => F.set(v, "nonesuch", 1));        // no such field
    return out;
  });
  check("a kind mismatch is refused", /kind mismatch/.test(errors[0] ?? ""), errors[0] ?? "accepted!");
  check("an unknown port names the real ones", /has no input/.test(errors[1] ?? ""), errors[1] ?? "accepted!");
  check("an unknown node type suggests types()", /unknown node type/.test(errors[2] ?? ""), errors[2] ?? "accepted!");
  check("an unknown field names the real ones", /has no field/.test(errors[3] ?? ""), errors[3] ?? "accepted!");
}

// --- a node the library refuses is reported, not dropped ---------------------------------------------
{
  // The return code of every ft_model_* call used to be summed and never looked at, so a constraint
  // the library rejected vanished from the model and the editor answered a different problem.
  const msg = await page.evaluate(() => {
    const F = window.ferrotherm;
    F.clear();
    const t = F.add("integer", 40, 40, { name: "temperature", lo: 10, hi: 20 });
    const other = F.add("integer", 40, 200, { name: "other", lo: 10, hi: 20 });
    const fix = F.add("fix", 300, 60, { value: 3 });   // 3 is a slot, not a temperature in 10..=20
    F.connect(t, fix, "var");
    const ne = F.add("notequal", 300, 200);
    F.connect(t, ne, "a"); F.connect(other, ne, "b");
    const s = F.add("solve", 600, 120);
    F.connect(fix, s); F.connect(ne, s);
    F.connect(s, F.add("report", 860, 120), "result");
    return F.run();
  });
  check("a refused constraint is reported by name and reason",
        /refused/.test(msg) && /10\.\.=20/.test(msg),
        msg.split("\n")[0].slice(0, 96));
  check("and the editor does not report an answer instead",
        !/feasible {2}yes/.test(msg));
}

// --- a spin variable speaks in -1 and +1 -------------------------------------------------------------
{
  const txt = await page.evaluate(() => {
    const F = window.ferrotherm;
    F.clear();
    const a = F.add("integer", 40, 40, { name: "a", lo: -1, hi: 1 });
    const fix = F.add("fix", 300, 40, { value: -1 });
    F.connect(a, fix, "var");
    const s = F.add("solve", 600, 40);
    F.connect(fix, s);
    F.connect(s, F.add("report", 860, 40), "result");
    return F.run();
  });
  check("a negative value survives the wasm boundary", /a = -1/.test(txt),
        txt.split("\n").find(l => l.includes("a =")) ?? txt.slice(0, 60));
}

// --- a term over three variables, which needed a whole compiler pass to be sayable -------------------
{
  const r = await page.evaluate(() => {
    const F = window.ferrotherm;
    F.clear();
    const vs = ["a", "b", "c"].map((n, i) => F.add("categorical", 40, 40 + i * 90,
                                                   { name: n, values: 3 }));
    const together = F.add("together", 320, 60, { value: 2, weight: 9, maximize: 1 });
    vs.forEach((v, i) => F.connect(v, together, `in ${i + 1}`));
    const s = F.add("solve", 640, 120);
    F.connect(together, s);
    F.connect(s, F.add("report", 900, 120), "result");
    return F.run();
  });
  const vals = Object.fromEntries([...r.matchAll(/(\w) = (\d)/g)].map(m => [m[1], m[2]]));
  check("three variables can be rewarded together",
        vals.a === "2" && vals.b === "2" && vals.c === "2",
        JSON.stringify(vals));
  check("and the report says what the lowering cost", /ancillas\s+[1-9]/.test(r),
        r.split("\n").find(l => l.includes("ancillas")) ?? "no ancilla line");
  check("with the caveat, not just the number", /optima rather than sampling/.test(r));
}

// --- a pairwise model reports none, so the number is a signal ----------------------------------------
{
  const r = await page.evaluate(() => {
    const F = window.ferrotherm;
    F.clear();
    const a = F.add("categorical", 40, 40, { name: "a", values: 3 });
    const b = F.add("categorical", 40, 160, { name: "b", values: 3 });
    const ne = F.add("notequal", 300, 60);
    F.connect(a, ne, "a"); F.connect(b, ne, "b");
    const s = F.add("solve", 600, 60);
    F.connect(ne, s);
    F.connect(s, F.add("report", 860, 60), "result");
    return F.run();
  });
  check("a pairwise model reports no ancillas", !/ancillas/.test(r));
}

// --- all_different: the constraint the editor could not state -------------------------------------
//
// Three regions of three colours, said once instead of as three pairwise inequalities. The point
// of the test is not that it solves -- pairwise already did -- it is that the ONE node means the
// same thing, and that asking it of more variables than there are values is refused when the model
// compiles rather than answered at some penalty.
{
  const distinct = n => page.evaluate(n => {
    const F = window.ferrotherm;
    F.clear();
    const vs = [];
    for (let i = 0; i < n; i++) vs.push(F.add("categorical", 40, 40 + i * 90, { name: "v" + i, values: 3 }));
    const ad = F.add("alldifferent", 320, 80);
    vs.forEach(v => F.connect(v, ad));
    const s = F.add("solve", 620, 80);
    F.connect(ad, s);
    F.connect(s, F.add("report", 900, 80), "result");
    return F.run();
  }, n);

  const three = await distinct(3);
  const got = [...three.matchAll(/v\d = (\d)/g)].map(m => m[1]);
  check("all different, in one node", new Set(got).size === 3 && three.includes("feasible  yes"),
        `three variables took ${JSON.stringify(got)}`);

  const four = await distinct(4);
  check("and four of them over three values is refused, not answered",
        /could not compile|refused/i.test(four) && !/feasible  yes/.test(four),
        four.split("\n").find(l => l.trim()) ?? "");
}

// --- exactly-one and at-most-one lower without slack ----------------------------------------------
{
  const one = (kind, k) => page.evaluate(([kind, k]) => {
    const F = window.ferrotherm;
    F.clear();
    const vs = [];
    for (let i = 0; i < 4; i++) vs.push(F.add("binary", 40, 40 + i * 90, { name: "v" + i }));
    const c = F.add(kind, 320, 60, k === null ? { value: 1 } : { k, value: 1 });
    vs.forEach(v => F.connect(v, c));
    const s = F.add("solve", 620, 200);
    F.connect(c, s);
    vs.forEach((v, i) => {
      const o = F.add("prefer", 320, 360 + i * 80, { value: 1, weight: 4 - i, maximize: 1 });
      F.connect(v, o, "var"); F.connect(o, s);
    });
    F.connect(s, F.add("report", 900, 200), "result");
    const txt = F.run();
    return { on: [...txt.matchAll(/v(\d) = 1/g)].length, spins: +txt.match(/(\d+) spins/)[1],
             feasible: txt.includes("feasible  yes") };
  }, [kind, k]);

  const eo = await one("exactlyone", null);
  const c1 = await one("cardinality", 1);
  check("exactly one takes exactly one", eo.feasible && eo.on === 1);
  // NOT cheaper than cardinality(k=1): the library's own comment says the two compile to identical
  // graphs, and the first version of this test asserted the opposite and failed. Where the saving
  // is real is against the INEQUALITY, which has to buy a slack variable to become an equality the
  // sampler can square, and at k = 1 the pairwise exclusion says the same thing for free.
  check("and it costs the same as the equality form at k=1", eo.spins === c1.spins,
        `${eo.spins} spins either way, as the compiler says`);

  const am1 = await one("atmost", 1);
  const amo1 = await one("atmostone", null);
  check("but at-most-one saves the inequality's slack", amo1.spins < am1.spins,
        `${amo1.spins} against ${am1.spins} for at_most with k = 1`);

  // The separator. With every variable rewarded, at-most-one still takes one; the difference from
  // exactly-one only shows when taking any is penalised, which is the test below.
  check("at most one is a ceiling, not a quota", amo1.feasible && amo1.on === 1,
        `took ${amo1.on} when all four were rewarded`);

  // And the library says so unprompted, which is the caveat channel the report now carries.
  const advice = await page.evaluate(() => {
    const F = window.ferrotherm;
    F.clear();
    const vs = [];
    for (let i = 0; i < 4; i++) vs.push(F.add("binary", 40, 40 + i * 90, { name: "v" + i }));
    const c = F.add("atmost", 320, 60, { k: 1, value: 1 });
    vs.forEach(v => F.connect(v, c));
    const s = F.add("solve", 620, 200);
    F.connect(c, s);
    F.connect(s, F.add("report", 900, 200), "result");
    return F.run();
  });
  check("and the compiler volunteers the cheaper form",
        /note: /.test(advice) && /at_most_one/.test(advice),
        (advice.split("\n").find(l => /note:/.test(l)) ?? "said nothing").trim().slice(0, 90));
}

// --- soft constraints: a preference the solver may trade away -------------------------------------
//
// This is the assertion that separates soft from hard, and it is about the WORD in the report, not
// about the values. Hard: the objective is outbid, the constraint holds, the answer is feasible.
// Soft at a price the objective beats: the constraint is broken, and the answer is STILL feasible,
// because a preference the modeller priced is not a rule.
{
  const priced = soft => page.evaluate(soft => {
    const F = window.ferrotherm;
    F.clear();
    const a = F.add("binary", 40, 40, { name: "a" });
    const b = F.add("binary", 40, 160, { name: "b" });
    const ne = F.add("notequal", 300, 60, { soft });
    F.connect(a, ne, "a"); F.connect(b, ne, "b");
    const s = F.add("solve", 620, 60);
    F.connect(ne, s);
    // Both rewarded for taking 1, which a ≠ b forbids. Something has to give.
    for (const v of [a, b]) {
      const o = F.add("prefer", 320, 300 + v * 60, { value: 1, weight: 20, maximize: 1 });
      F.connect(v, o, "var"); F.connect(o, s);
    }
    F.connect(s, F.add("report", 900, 60), "result");
    const txt = F.run();
    return { txt, both: /a = 1/.test(txt) && /b = 1/.test(txt),
             feasible: txt.includes("feasible  yes") };
  }, soft);

  const hard = await priced(0);
  check("a hard constraint outbids the objective", hard.feasible && !hard.both,
        "a rule is a rule at any weight");

  const soft = await priced(1);
  check("a soft one is bought out", soft.both, "both took the rewarded value");
  check("and the answer is still feasible", soft.feasible,
        "a traded preference is not an infeasibility");
  check("the report says traded, not broken", /traded:/.test(soft.txt) && !/broken:/.test(soft.txt),
        (soft.txt.split("\n").find(l => /traded|broken/.test(l)) ?? "said neither").trim());
  check("and prices what the trade cost", /traded\s+[\d.]+ of preference/.test(soft.txt));
}

// --- encodings are a choice, and an inexact one says so --------------------------------------------
{
  // Two models: one variable pinned, and two variables under a pairwise constraint. The encoding
  // wins the first and loses the second, and only measuring both says so.
  const alone = encoding => page.evaluate(encoding => {
    const F = window.ferrotherm;
    F.clear();
    const a = F.add("categorical", 40, 40, { name: "a", values: 4, encoding });
    const fx = F.add("fix", 300, 40, { value: 1 });
    F.connect(a, fx, "var");
    const s = F.add("solve", 620, 60);
    F.connect(fx, s);
    F.connect(s, F.add("report", 900, 60), "result");
    const txt = F.run();
    return { txt, spins: +(txt.match(/(\d+) spins/)?.[1] ?? -1) };
  }, encoding);

  const paired = encoding => page.evaluate(encoding => {
    const F = window.ferrotherm;
    F.clear();
    const a = F.add("categorical", 40, 40, { name: "a", values: 4, encoding });
    const b = F.add("categorical", 40, 200, { name: "b", values: 4, encoding });
    const ne = F.add("notequal", 300, 60);
    F.connect(a, ne, "a"); F.connect(b, ne, "b");
    const s = F.add("solve", 620, 60);
    F.connect(ne, s);
    F.connect(s, F.add("report", 900, 60), "result");
    const txt = F.run();
    return { txt, spins: +(txt.match(/(\d+) spins/)?.[1] ?? -1) };
  }, encoding);

  const h1 = await alone("one-hot"), w1 = await alone("domain-wall");
  check("domain-wall saves a spin on the variable itself", w1.spins === h1.spins - 1,
        `${w1.spins} against ${h1.spins} for one 4-value variable`);

  const h2 = await paired("one-hot"), w2 = await paired("domain-wall");
  check("and loses it back across a pairwise constraint", w2.spins > h2.spins,
        `${w2.spins} against ${h2.spins}: the lowering pays ancillas to return to quadratic`);
  check("both encodings still answer", /feasible  yes/.test(w2.txt) && /feasible  yes/.test(h2.txt));

  // Binary is the third encoding the library has and the one the picker does not offer, because
  // the compiler refuses it in any constraint or objective. The test is that the editor cannot
  // reach it at all, rather than reaching it and handing back a refusal.
  const offered = await page.evaluate(() =>
    window.ferrotherm.types().categorical.fields.encoding);
  check("binary is not offered, since every model using it is refused",
        typeof offered === "string" && offered === "one-hot");
}

// --- a counting constraint over more than four variables ------------------------------------------
//
// The old node drew four ports, so this model could not be stated in the editor at all.
{
  const r = await page.evaluate(() => {
    const F = window.ferrotherm;
    F.clear();
    const vs = [];
    for (let i = 0; i < 9; i++) vs.push(F.add("binary", 40, 20 + i * 60, { name: "shift" + i }));
    const c = F.add("atmost", 320, 60, { k: 2, value: 1 });
    vs.forEach(v => F.connect(v, c));
    const s = F.add("solve", 620, 200);
    F.connect(c, s);
    vs.forEach((v, i) => {
      const o = F.add("prefer", 320, 620 + i * 40, { value: 1, weight: 9 - i, maximize: 1 });
      F.connect(v, o, "var"); F.connect(o, s);
    });
    F.connect(s, F.add("report", 900, 200), "result");
    const txt = F.run();
    return { on: [...txt.matchAll(/shift(\d) = 1/g)].map(m => +m[1]),
             feasible: txt.includes("feasible  yes") };
  });
  check("at most two of nine shifts", r.feasible && r.on.length === 2,
        `took ${JSON.stringify(r.on)}, which four fixed ports could not have said`);
  check("and it took the two worth most", r.on.join() === "0,1");
}

// --- every constraint the model layer has is reachable from the palette ---------------------------
//
// The gate that would have caught the gap this batch closed: the editor had six of nine, and
// nothing anywhere compared the two lists.
{
  const kinds = await page.evaluate(() => Object.entries(window.ferrotherm.types())
    .filter(([, T]) => T.kind === "cons").map(([k]) => k).sort());
  const want = ["alldifferent", "atleast", "atmost", "atmostone", "cardinality", "equal",
                "exactlyone", "fix", "notequal"];
  check("nine constraints, all of them", kinds.join() === want.join(), kinds.join(" "));
  const soft = await page.evaluate(() => Object.entries(window.ferrotherm.types())
    .filter(([, T]) => T.kind === "cons").every(([, T]) => "soft" in T.fields));
  check("and every one of them can be priced", soft, "soft is not a property of some constraints");
}

// --- the round trip: what an agent writes, a person can edit, and back -----------------------------
//
// The editor speaks in nodes, the HTTP and MCP APIs speak in variables and constraints. They are
// the same model said twice, and the assertion is that nothing is lost either way -- because a
// bridge that drops the objective, or the k, or a soft price, is worse than no bridge: it hands
// back a model that runs and means something else.
{
  const MODEL = {
    variables: [
      { name: "mon", values: 3 },
      { name: "tue", values: 3 },
      { name: "wed", values: 3 },
      { name: "spare", lo: 0, hi: 4 },
    ],
    constraints: [
      { type: "all_different", of: [{ var: "mon" }, { var: "tue" }, { var: "wed" }] },
      { type: "fix", var: "spare", value: 2 },
      { type: "not_equal", a: "mon", b: "spare", soft: 3 },
      { type: "at_most", k: 2, of: [{ var: "mon", value: 1 }, { var: "tue", value: 1 },
                                    { var: "wed", value: 1 }] },
    ],
    objective: { maximize: true, terms: [{ var: "mon", value: 2, weight: 5 }] },
  };

  const round = await page.evaluate(m => {
    const F = window.ferrotherm;
    const laid = F.fromModel(m);
    const back = F.toModel();
    return { laid, back, txt: F.run() };
  }, MODEL);

  check("a model an agent wrote lays out as a graph",
        round.laid.variables === 4 && round.laid.constraints === 4 && round.laid.terms === 1,
        JSON.stringify(round.laid));
  check("and the laid-out graph runs", /feasible  yes/.test(round.txt),
        (round.txt.split("\n").find(l => /feasible/.test(l)) ?? "").trim());

  const b = round.back;
  check("every variable survives the trip",
        JSON.stringify(b.variables.map(v => v.name).sort()) ===
        JSON.stringify(["mon", "spare", "tue", "wed"]), JSON.stringify(b.variables));
  check("an integer keeps its range, not a value count",
        b.variables.some(v => v.name === "spare" && v.lo === 0 && v.hi === 4));
  check("every constraint survives, by type",
        JSON.stringify(b.constraints.map(c => c.type).sort()) ===
        JSON.stringify(["all_different", "at_most", "fix", "not_equal"]),
        JSON.stringify(b.constraints.map(c => c.type)));
  check("k survives", b.constraints.find(c => c.type === "at_most")?.k === 2);
  // The one most likely to be dropped silently, because a graph that loses it still draws and
  // still runs -- it just answers a different question.
  check("and so does the soft price", b.constraints.find(c => c.type === "not_equal")?.soft === 3,
        JSON.stringify(b.constraints.find(c => c.type === "not_equal")));
  check("the objective survives with its sense and weight",
        b.objective?.maximize === true && b.objective.terms[0].weight === 5 &&
        b.objective.terms[0].var === "mon" && b.objective.terms[0].value === 2,
        JSON.stringify(b.objective));

  // Second trip. A conversion that is stable is a conversion; one that drifts is a translation.
  const twice = await page.evaluate(m => {
    const F = window.ferrotherm;
    F.fromModel(m);
    const once = F.toModel();
    F.fromModel(once);
    return { once, twice: F.toModel() };
  }, MODEL);
  const norm = m => JSON.stringify(m, Object.keys(m).sort());
  check("and a second trip changes nothing",
        JSON.stringify(twice.once) === JSON.stringify(twice.twice),
        "the mapping is stable, not merely reversible once");

  // What it refuses. A counting constraint whose literals name different values has no drawing,
  // and saying so is better than drawing the one it can and meaning something else.
  const undrawable = await page.evaluate(() => {
    try {
      window.ferrotherm.fromModel({
        variables: [{ name: "a", values: 3 }, { name: "b", values: 3 }],
        constraints: [{ type: "at_most", k: 1,
                        of: [{ var: "a", value: 0 }, { var: "b", value: 2 }] }],
      });
      return "drew it anyway";
    } catch (e) { return e.message; }
  });
  check("a model with no drawing is refused, not approximated",
        /has no drawing/.test(undrawable), undrawable.slice(0, 90));
  check("and the refusal says where it still runs", /HTTP and MCP/.test(undrawable));

  const undeclared = await page.evaluate(() => {
    try {
      window.ferrotherm.fromModel({ variables: [{ name: "a", values: 3 }],
                                    constraints: [{ type: "equal", a: "a", b: "ghost" }] });
      return "accepted a ghost";
    } catch (e) { return e.message; }
  });
  check("a constraint naming an undeclared variable is refused by name",
        /"ghost".*never declares/.test(undeclared), undeclared.slice(0, 80));
}

// --- a model in a link -----------------------------------------------------------------------------
//
// The last leg: an agent hands a person the model itself rather than a description of it. The test
// is that following the link produces the model, and that a link which does not decode opens the
// example with an explanation instead of a blank page or a stack trace.
{
  const MODEL = {
    variables: [{ name: "red", values: 4 }, { name: "blue", values: 4 }, { name: "hours", lo: 1, hi: 9 }],
    constraints: [{ type: "not_equal", a: "red", b: "blue" },
                  { type: "fix", var: "hours", value: 6, soft: 2 }],
    objective: { maximize: true, terms: [{ var: "red", value: 3, weight: 7 }] },
  };
  const url = await page.evaluate(m => {
    window.ferrotherm.fromModel(m);
    return window.ferrotherm.link();
  }, MODEL);
  check("a graph becomes a link", /#model=[A-Za-z0-9_-]+$/.test(url), url.slice(0, 60) + "...");

  const fresh = await browser.newPage();
  const freshErrs = [];
  fresh.on("pageerror", e => freshErrs.push(e.message));
  await fresh.goto(url);
  await fresh.evaluate(() => window.ferrotherm.ready);
  const back = await fresh.evaluate(() => window.ferrotherm.toModel());
  check("and following it produces the model, not the example",
        JSON.stringify(back.variables) === JSON.stringify(MODEL.variables)
        && back.constraints.length === 2
        && back.objective?.terms[0].weight === 7,
        JSON.stringify(back.variables));
  check("the soft price rides in the link too",
        back.constraints.find(c => c.type === "fix")?.soft === 2);
  const linkedRun = await fresh.evaluate(() => window.ferrotherm.run());
  check("and the linked model runs", /feasible  yes/.test(linkedRun));

  // The fragment is the point: it is the half of a URL browsers do not send to a server, so
  // opening somebody's model does not put their problem in a log.
  check("the model is in the fragment, not the query", !url.includes("?"),
        "a query string would reach the server");

  // Changing only the fragment does not reload a page, so the editor listens for it: pasting a
  // model link into an already-open editor used to do nothing at all, which reads as a broken link
  // rather than as a page that never looked.
  const second = { variables: [{ name: "solo", values: 5 }, { name: "duo", values: 5 }],
                   constraints: [{ type: "not_equal", a: "solo", b: "duo" }] };
  const secondUrl = await page.evaluate(m => {
    window.ferrotherm.fromModel(m);
    return window.ferrotherm.link();
  }, second);
  await fresh.evaluate(u => { location.hash = new URL(u).hash; }, secondUrl);
  const swapped = await fresh.evaluate(() => window.ferrotherm.toModel());
  check("a link pasted into an open editor is followed",
        swapped.variables.length === 2 && swapped.variables[0].name === "solo",
        JSON.stringify(swapped.variables));

  // A variable nothing mentions is not reachable from Solve, so toModel would drop it. Refused on
  // the way IN, by name, rather than lost on the way out.
  const orphan = await page.evaluate(() => {
    try {
      window.ferrotherm.fromModel({
        variables: [{ name: "a", values: 3 }, { name: "unused", values: 3 }],
        constraints: [{ type: "fix", var: "a", value: 1 }],
      });
      return "drew it and lost the variable";
    } catch (e) { return e.message; }
  });
  check("a variable nothing mentions is refused by name, not silently dropped",
        /"unused"/.test(orphan) && /lost on the way back out/.test(orphan), orphan.slice(0, 80));

  // A link that does not decode is not a crash and not a blank page.
  await fresh.goto(url.split("#")[0] + "#model=not-base64-at-all!!");
  await fresh.reload();
  await fresh.evaluate(() => window.ferrotherm.ready);
  const fallback = await fresh.evaluate(() => document.getElementById("out").textContent);
  const nodesThere = await fresh.evaluate(() => window.ferrotherm.nodes.length);
  check("a link that does not decode falls back to the example",
        nodesThere > 0 && /Graph colouring/.test(fallback),
        "not a blank page");
  check("and no page error escapes", freshErrs.length === 0, freshErrs.join(" | "));
  await fresh.close();
}

// --- an answer is scored in the modeller's own units -----------------------------------------------
//
// The report used to carry ONE number: the compiled Ising energy, with every penalty and the
// constant folded in. A person who wires up "Prefer value" nodes cannot read what their answer is
// worth out of that, cannot compare two answers by it, and cannot tell a good answer from a barely
// feasible one -- and it moves when the penalty does, so it is not even stable across edits.
{
  const txt = await page.evaluate(() => {
    const F = window.ferrotherm;
    F.clear();
    const mon = F.add("categorical", 40, 40, { name: "mon", values: 3 });
    const tue = F.add("categorical", 40, 200, { name: "tue", values: 3 });
    const ne = F.add("notequal", 300, 60);
    F.connect(mon, ne, "a"); F.connect(tue, ne, "b");
    const s = F.add("solve", 620, 60);
    F.connect(ne, s);
    const o1 = F.add("prefer", 320, 320, { value: 1, weight: 5, maximize: 1 });
    F.connect(mon, o1, "var"); F.connect(o1, s);
    const o2 = F.add("prefer", 320, 420, { value: 2, weight: 4, maximize: 1 });
    F.connect(tue, o2, "var"); F.connect(o2, s);
    F.connect(s, F.add("report", 900, 60), "result");
    return F.run();
  });
  const obj = +(txt.match(/objective\s+(-?[\d.]+)/)?.[1] ?? NaN);
  const energy = +(txt.match(/energy\s+(-?[\d.]+)/)?.[1] ?? NaN);
  check("the report says what the answer is worth", obj === 9,
        `objective ${obj}; the optimum of 5*[mon=1] + 4*[tue=2] under mon != tue is 9`);
  check("and labels which number is which", /your units/.test(txt) && /compiled Ising/.test(txt));
  check("the two numbers are not the same one", obj !== energy,
        `objective ${obj}, energy ${energy}`);
}

// --- drawing a machine and fitting it ----------------------------------------------------------------
//
// Every other graph in this editor DESCRIBES a model. A Train chain hands the sampler data and gets
// a model back, which is the only path here that produces one. The gates that pass on this page
// check that a symbol is REACHABLE; this drives the chain a person would draw.
{
  const wide = await page.evaluate(() => {
    const F = window.ferrotherm;
    F.clear();
    const d = F.add("dataset", 40, 300, { images: "bars-and-stripes-3" });
    const h = F.add("hidden", 300, 160, { units: 12 });
    F.connect(d, h, "below");
    const t = F.add("train", 580, 60, { epochs: 400, k: 10, seed: 3 });
    F.connect(h, t, "top");
    F.connect(t, F.add("report", 840, 60), "result");
    return F.run();
  });
  const pct = +(wide.match(/learned\s+([\d.]+)%/)?.[1] ?? NaN);
  const before = +(wide.match(/before fitting\s+(-?[\d.]+)/)?.[1] ?? NaN);
  check("a drawn machine fits", pct > 85, `learned ${pct}%; a wide machine reaches the nineties`);
  // Derived, not measured: every weight starts at zero, so the machine is uniform over 2^9 images.
  check("the untrained end of the scale is exact", Math.abs(before + 9 * Math.LN2) < 5e-4,
        `before ${before}, and -9 ln 2 is ${(-9 * Math.LN2).toFixed(4)}`);
  check("the machine's shape is reported", /9 - 12/.test(wide) && /21 spins/.test(wide), wide.slice(0, 120));

  // Depth is drawn as chain length, so a deeper chain is a different machine and must score lower
  // at the same latent count -- the ordering this repository measured.
  const deep = await page.evaluate(() => {
    const F = window.ferrotherm;
    F.clear();
    const d = F.add("dataset", 40, 420, { images: "bars-and-stripes-3" });
    const h1 = F.add("hidden", 300, 300, { units: 6 });
    const h2 = F.add("hidden", 300, 160, { units: 6 });
    F.connect(d, h1, "below"); F.connect(h1, h2, "below");
    const t = F.add("train", 580, 60, { epochs: 400, k: 10, seed: 3 });
    F.connect(h2, t, "top");
    F.connect(t, F.add("report", 840, 60), "result");
    return F.run();
  });
  const deepPct = +(deep.match(/learned\s+([\d.]+)%/)?.[1] ?? NaN);
  check("stacking layers is a different machine", /9 - 6 - 6/.test(deep), deep.slice(0, 120));
  check("and the same latents learn less when stacked", deepPct < pct,
        `one layer of 12 reached ${pct}%, two of 6 reached ${deepPct}%`);

  // The picture and the JSON are one document, which is a claim this page makes in llms.txt and
  // which a machine would quietly falsify if toModel only knew about problems.
  const round = await page.evaluate(() => {
    const F = window.ferrotherm;
    const m = F.toModel();
    F.clear();
    F.fromModel(m);
    return { m, back: F.toModel(), types: F.nodes.map(n => n.type).sort() };
  });
  check("a machine round-trips through the model", JSON.stringify(round.m) === JSON.stringify(round.back),
        JSON.stringify(round.m) + " vs " + JSON.stringify(round.back));
  check("and comes back as the chain a person drew",
        round.types.join(",") === "dataset,hidden,hidden,report,train", round.types.join(","));

  // A chain with no data is refused before anything runs, and it names the NODE and the PORT rather
  // than the chain -- "Hidden layer#7: unwired input: below" points at what to fix, where "this
  // chain does not reach a Dataset" would leave a person hunting for which link is missing.
  const bad = await page.evaluate(() => {
    const F = window.ferrotherm;
    F.clear();
    const t = F.add("train", 580, 60);
    F.connect(F.add("hidden", 300, 160, { units: 4 }), t, "top");
    F.connect(t, F.add("report", 840, 60), "result");
    return F.run();
  });
  check("a chain with no data is refused before it runs",
        /fix these first/.test(bad) && /Hidden layer/.test(bad) && /below/.test(bad),
        bad.slice(0, 140));
  check("and a Train with nothing wired in says which port",
        /wire a Hidden layer in/.test(await page.evaluate(() => {
          const F = window.ferrotherm;
          F.clear();
          const t = F.add("train", 580, 60);
          F.connect(t, F.add("report", 840, 60), "result");
          return F.run();
        })));
}

// --- nothing threw along the way ---------------------------------------------------------------------
check("no page errors", errs.length === 0, errs.join(" | "));

await browser.close();
server.close();
console.log(failed ? `\n${failed} failed` : "\nall passed");
process.exit(failed ? 1 : 0);
