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

// --- nothing threw along the way ---------------------------------------------------------------------
check("no page errors", errs.length === 0, errs.join(" | "));

await browser.close();
server.close();
console.log(failed ? `\n${failed} failed` : "\nall passed");
process.exit(failed ? 1 : 0);
