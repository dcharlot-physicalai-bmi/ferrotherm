// Solve the agreed model through the WASM build, in a real browser engine.
//
// `scripts/sem-wasm.mjs` drives the editor's UI to read a compiled `.ftp`. This asks the harder
// question -- does the wasm return the same ANSWER as the other surfaces -- and asks it of the
// binary itself rather than the page around it: it instantiates `docs/ferrotherm.wasm` fresh and
// calls the C ABI directly. The page's layout can change without that meaning the sampler drifted,
// and the sampler can drift without the layout changing.
//
// `ferrotherm.wasm` takes no imports, so instantiating it needs nothing but a fetch.
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
  try {
    const b = await readFile(p);
    res.writeHead(200, { "content-type": MIME[extname(p)] ?? "application/octet-stream" });
    res.end(b);
  } catch { res.writeHead(404).end("nope"); }
});
await new Promise((r) => server.listen(0, r));
const base = `http://localhost:${server.address().port}`;

const b = await chromium.launch();
const p = await b.newPage();
await p.goto(base + "/ide.html");

const out = await p.evaluate(async () => {
  const r = await WebAssembly.instantiateStreaming(fetch("ferrotherm.wasm"), {});
  const W = r.instance.exports;
  const m = W.ft_model_new();

  // The same model every other surface solves: two categoricals, an integer, not_equal, a
  // counting constraint, fix, and a two-term objective. Unique optimum a=1, b=2, t=12.
  const a = W.ft_model_categorical(m, 3);
  const bv = W.ft_model_categorical(m, 3);
  const t = W.ft_model_integer(m, 10n, 13n);
  W.ft_model_not_equal(m, a, bv);

  // at_most 1 of {a=0, b=0}, through the literal list the C ABI takes.
  // Build the pending literal list, then close it with kind 1 = "at most k".
  W.ft_model_lits_clear(m);
  W.ft_model_lit(m, a, 0n);
  W.ft_model_lit(m, bv, 0n);
  if (!W.ft_model_close(m, 1, 1)) return "at_most refused";

  W.ft_model_fix(m, t, 12n);
  W.ft_model_objective_term(m, 1, 3.0, a, 1n);
  W.ft_model_objective_term(m, 1, 4.0, bv, 2n);

  if (!W.ft_model_compile(m)) return "compile refused";
  if (!W.ft_model_solve(m, 64)) return "solve refused";

  const val = (i) => {
    const got = W.ft_model_value(m, i);
    return got === -9223372036854775808n ? "null" : got.toString();
  };
  return `a=${val(a)} b=${val(bv)} t=${val(t)} feasible=${W.ft_model_feasible(m) === 1}`;
});

process.stdout.write(out);
await b.close();
server.close();
