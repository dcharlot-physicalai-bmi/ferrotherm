// Fit a model to data through the WASM build, and print what it learned.
//
// The browser is the surface where fitting is newest and least watched: the C, Python, Zig and
// Julia arms each have a test, and until this file the wasm arm had a page nobody could assert
// against. `check-fit.sh` runs this beside the native library and requires the two to AGREE.
//
// No playwright, no server, no DOM. `ferrotherm.wasm` takes no imports, so node instantiates it
// directly -- which also means this checks THE BINARY THE PAGE LOADS rather than a rebuild of it,
// and a stale committed wasm fails here instead of silently serving an older sampler.
import { readFile } from "node:fs/promises";

const here = new URL(".", import.meta.url).pathname;
const arg = (name, fallback) => {
  const i = process.argv.indexOf(name);
  return i < 0 ? fallback : process.argv[i + 1];
};
const seed = BigInt(arg("--seed", "3"));
const wasm = arg("--wasm", `${here}../docs/ferrotherm.wasm`);

const { instance } = await WebAssembly.instantiate(await readFile(wasm), {});
const W = instance.exports;

const ebmError = () => {
  const need = W.ft_ebm_error(0, 0);
  if (!need) return "";
  const p = W.ft_scratch(need);
  const got = W.ft_ebm_error(p, need);
  return new TextDecoder().decode(new Uint8Array(W.memory.buffer, p, got));
};
const die = (why) => { console.error(why); process.exit(2); };

// The dataset, read out of the library rather than written here: a benchmark the checker and the
// library disagree about is a benchmark that proves nothing.
const side = 3, visible = side * side;
const rowCount = W.ft_ebm_bars_and_stripes(side, 0, 0);
if (!rowCount) die("bars and stripes: " + ebmError());
const need = rowCount * visible;
const p0 = W.ft_scratch(need);
if (!W.ft_ebm_bars_and_stripes(side, p0, need)) die("bars and stripes: " + ebmError());
// Copy out: ft_scratch is one shared buffer and the next call overwrites it.
const rows = Array.from(new Int8Array(W.memory.buffer, p0, need));
const push = () => {
  const q = W.ft_scratch(rows.length);
  new Int8Array(W.memory.buffer, q, rows.length).set(rows);
  return q;
};

/** Fit one machine and report the two ends of the scale plus what it reached. */
function fit(makeSim, label) {
  const sim = makeSim();
  if (!sim) die(label + ": " + ebmError());
  const before = W.ft_ebm_log_likelihood(sim, visible, push(), rowCount);
  if (!W.ft_ebm_train(sim, visible, push(), rowCount, 400, 10, 0, 0, 0, seed)) {
    die(label + ": " + ebmError());
  }
  const after = W.ft_ebm_log_likelihood(sim, visible, push(), rowCount);
  const spins = W.ft_len(sim);
  W.ft_free(sim);
  const floor = -visible * Math.LN2, ceiling = -Math.log(rowCount);
  return { label, spins, before, after, learned: (after - floor) / (ceiling - floor) * 100 };
}

const wide = fit(() => W.ft_ebm_rbm(visible, 12, 1.0, 1n), "wide");

// The deep arm exists here to exercise the u32 ARRAY MARSHALLING, which is the one thing in the
// page's fitting path that a scalar call cannot reach. ft_scratch hands back a Vec<u8> pointer with
// no 4-byte alignment guarantee, so the page writes the widths through a DataView; a Uint32Array
// view on an unaligned offset throws, and it would throw in a click handler where nobody sees it.
const deep = fit(() => {
  const q = W.ft_scratch(8);
  const dv = new DataView(W.memory.buffer, q, 8);
  dv.setUint32(0, 6, true);
  dv.setUint32(4, 6, true);
  return W.ft_ebm_dbm(visible, q, 2, 1.0, 1n);
}, "deep");

console.log(JSON.stringify({ rows: rowCount, visible, wide, deep }, null, 2));
