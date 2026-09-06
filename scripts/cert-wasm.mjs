// CERTIFY one graph through the WASM build, in a real browser engine.
//
// `scripts/ans-wasm.mjs` asks whether the browser returns the same ANSWER. This asks the question
// the roadmap's acceptance criterion is actually written in: does it sample the same
// DISTRIBUTION. An optimum is one state and a distribution is all of them, so a sampler can agree
// about the first while disagreeing about the second -- and the certificate is the only instrument
// in this stack that can tell the difference.
//
// Not a rerun of the native check in a different process. `exp` and `ln` come from the platform's
// libm, and wasm32's need not agree with aarch64's in the last ulp; one ulp changes an acceptance,
// which changes a chain. So the two are compared the way two measurements are compared -- against
// the certificate's own sampling-noise floor -- rather than for equality.
//
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

  // `ising::ring(10, 1.0, 0.3)`, built through the ABI because the browser has no `ft_ring`:
  // couple i to i+1 around the loop, and bias every site. Ten spins, so the exact Boltzmann
  // distribution is enumerable and `tv` is a real distance rather than an estimate.
  const N = 10;
  const b = W.ft_builder_new(N);
  for (let i = 0; i < N; i++) {
    if (!W.ft_builder_couple(b, i, (i + 1) % N, 1.0)) return { error: `couple ${i} refused` };
    if (!W.ft_builder_bias(b, i, 0.3)) return { error: `bias ${i} refused` };
  }
  const sim = W.ft_builder_build(b, 0.5, 11n);
  if (!sim) return { error: "builder produced no simulation" };
  if (!W.ft_certify(sim, 3000, 8)) return { error: "certify refused" };

  const findings = [];
  for (let i = 0; i < W.ft_cert_findings(sim); i++) findings.push(i);

  const got = {
    beta_eff: W.ft_cert_beta_eff(sim),
    beta_lo: W.ft_cert_beta_lo(sim),
    beta_hi: W.ft_cert_beta_hi(sim),
    tau: W.ft_cert_tau(sim),
    ess: W.ft_cert_ess(sim),
    tv: W.ft_cert_tv(sim),
    floor: W.ft_cert_floor(sim),
    passed: W.ft_cert_passed(sim) === 1,
    findings: findings.length,
  };
  W.ft_free(sim);
  return got;
});

await b.close();
server.close();

if (out.error) {
  console.error(`the wasm build could not certify: ${out.error}`);
  process.exit(1);
}
console.log(JSON.stringify(out));
