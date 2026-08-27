// Drive the DEPLOYED workbench through the paths this release added. Byte-identical to the repo is
// not the same as running: the wasm and the page are separate fetches, and a page that loads with a
// stale module answers questions in the old vocabulary with nothing to say so.
import { chromium } from 'playwright';
const URL = 'https://energy.physicalai-bmi.org/assets/ferrotherm/ide.html';
const b = await chromium.launch(); const p = await b.newPage();
const errs = [];
p.on('pageerror', e => errs.push('pageerror: ' + e.message));
p.on('console', m => { if (m.type() === 'error') errs.push('console: ' + m.text()); });
await p.goto(URL, { waitUntil: 'load' });
await p.waitForFunction(() => document.getElementById('status')?.textContent
  && !document.getElementById('status').textContent.includes('loading'), null, { timeout: 30000 });

let fail = 0;
const check = (n, ok, d = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${n}${d ? '   ' + d : ''}`); if (!ok) fail++; };

const run = (json) => p.evaluate((j) => {
  const el = document.getElementById('src');
  el.value = j; el.dispatchEvent(new Event('input', { bubbles: true }));
  if (typeof window.apply === 'function') window.apply(); else document.getElementById('apply')?.click();
  return document.getElementById('status').textContent;
}, json);

const body = (extra) => JSON.stringify({
  variables: [{ name: 'a', values: 2 }, { name: 'b', values: 2 }],
  constraints: [{ type: 'not_equal', a: 'a', b: 'b', ...extra }],
  objective: { maximize: true, terms: [{ var: 'a', value: 0, weight: 5 }, { var: 'b', value: 0, weight: 5 }] },
  tries: 24,
});

check('the live editor trades a priced preference', /preference\(s\) traded for 1\.00/.test(await run(body({ soft: 1 }))));
check('and keeps one priced above the objective', /every constraint holds/.test(await run(body({ soft: 50 }))));
check('and refuses a price that is not a number', /is a price/.test(await run(body({ soft: '5' }))));

await run(JSON.stringify({ graph: { builtin: 'lattice2d', l: 4, j: 1.0 }, beta: 0.44, seed: 1 }));
const probe = (s) => p.evaluate((st) => {
  try { return { ok: true, e: window.__putSpins(Int32Array.from(st)) }; }
  catch (err) { return { ok: false, m: err.message }; }
}, s);
const good = await probe(Array(16).fill(1));
check('the live GPU-readback guard accepts a real state', good.ok && good.e === -32, JSON.stringify(good));
const short = await probe(Array(13).fill(1));
check('and refuses a short readback instead of padding it', !short.ok && /did not complete/.test(short.m), JSON.stringify(short));

// The optimality bracket, against the DEPLOYED page and the DEPLOYED wasm. This is the check that
// would have caught the stale module: `ft_bound_sdp` missing makes the call `undefined`, and calling
// undefined in a click handler leaves the page looking fine and doing nothing.
{
  const ring = { n: 12, couplings: [] };
  for (let i = 0; i < 12; i++) ring.couplings.push([i, (i + 1) % 12, i === 0 ? -1 : 1]);
  await run(JSON.stringify({ graph: ring, beta: 1, seed: 1 }));
  const r = await p.evaluate(() => {
    document.getElementById('method').value = 'branch';
    document.getElementById('solve').click();
    return {
      status: document.getElementById('status').textContent,
      verdict: document.getElementById('certverdict').textContent,
      shown: document.getElementById('cert').hidden === false,
      best: document.querySelector('#btab tr.best')?.cells[0].textContent ?? null,
      rows: [...document.querySelectorAll('#btab tr')].map(x => x.cells[0].textContent),
    };
  });
  check('the live workbench proves a small instance', /tree exhausted/.test(r.status), r.status.slice(0, 90));
  check('and shows the optimality bracket', r.shown && /PROVED OPTIMAL/.test(r.verdict), r.verdict);
  check('the live wasm really carries the SDP bound', r.rows.includes('sdp'), r.rows.join(','));
  const bl = await p.evaluate(() => {
    document.getElementById('method').value = 'bls';
    document.getElementById('solve').click();
    return document.getElementById('status').textContent;
  });
  check('and breakout local search runs on the deployed page', /descents/.test(bl), bl.slice(0, 90));
  check('and a bound is named as best', r.best !== null, String(r.best));
}

check('no page errors', errs.length === 0, errs.join(' | '));
await b.close();
console.log(fail ? `\n${fail} failed` : '\nall passed against the deployed page');
process.exit(fail ? 1 : 0);
