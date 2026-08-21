// The editor and the wasm are the same surface: the page builds the model through the C ABI
// compiled to wasm. Drive it and take the compiled program out of the DOM.
// FINDING PLAYWRIGHT, which is not where a `cd` can put it.
//
// CI installs playwright into `web-tests/node_modules` and nowhere else. A bare `import "playwright"`
// from a file in `scripts/` resolves against THE IMPORTING FILE's directory -- scripts/node_modules,
// then the repo root, then /. It never looks in a sibling directory, and no amount of changing the
// working directory moves it: cwd plays no part in bare-specifier resolution. An earlier fix ran
// this script from `web-tests` for exactly that reason and did nothing, and passed its local check
// only because a stray root `node_modules` was answering the import in both arrangements.
//
// So resolve it explicitly, against each place it could legitimately live, and say so when it is in
// none of them. `--probe` exits 0/1 without doing any work, so the shell gate can ask THIS question
// rather than a proxy for it -- the guard that decides must be the one the code will actually run.
import { createRequire } from "node:module";
const REPO = new URL("../", import.meta.url).pathname;
const require = createRequire(import.meta.url);
const PW_PATHS = [REPO, `${REPO}web-tests/`];
function loadPlaywright() {
  for (const base of PW_PATHS) {
    try {
      return require(require.resolve("playwright", { paths: [base] }));
    } catch (e) {
      if (e.code !== "MODULE_NOT_FOUND") throw e;
    }
  }
  throw new Error(
    `playwright resolves from none of: ${PW_PATHS.map((p) => `${p}node_modules`).join(", ")}`,
  );
}
if (process.argv.includes("--probe")) {
  try { loadPlaywright(); process.exit(0); } catch (e) { console.error(e.message); process.exit(1); }
}
const { chromium } = loadPlaywright();
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
