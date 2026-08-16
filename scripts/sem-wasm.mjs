// The editor and the wasm are the same surface: the page builds the model through the C ABI
// compiled to wasm. Drive it and take the compiled program out of the DOM.
import { chromium } from "playwright";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join } from "node:path";
const ROOT = new URL("../docs/", import.meta.url).pathname;
const MIME = { ".html": "text/html", ".wasm": "application/wasm", ".js": "text/javascript" };
const server = createServer(async (req, res) => {
  const p = join(ROOT, (req.url === "/" ? "/ide.html" : req.url).split("?")[0]);
  try { const b = await readFile(p); res.writeHead(200, { "content-type": MIME[extname(p)] ?? "application/octet-stream" }); res.end(b); }
  catch { res.writeHead(404).end("nope"); }
});
await new Promise((r) => server.listen(0, r));
const base = `http://localhost:${server.address().port}`;
const b = await chromium.launch(); const p = await b.newPage();
await p.goto(base + "/ide.html");
await p.waitForFunction(() => document.getElementById("status")?.textContent
  && !document.getElementById("status").textContent.includes("loading"), null, { timeout: 30000 });
const model = await readFile("/tmp/_sem_model.json", "utf8");
const ftp = await p.evaluate((j) => {
  const el = document.getElementById("src");
  el.value = j; el.dispatchEvent(new Event("input", { bubbles: true }));
  if (typeof window.apply === "function") window.apply(); else document.getElementById("apply")?.click();
  const f = document.getElementById("ftp");
  return f.dataset.full ?? f.textContent;
}, model);
process.stdout.write(ftp);
await b.close(); server.close();
